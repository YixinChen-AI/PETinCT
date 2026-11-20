"""Full-resolution CT-to-PET FCN without spatial resampling.

This architecture is a controlled alternative to the MAISI/AVAE backbone.
Every feature tensor remains on the input voxel grid: there is no pooling,
strided convolution, interpolation, transposed convolution, or latent random
variable.  Receptive field is obtained with depthwise dilated convolutions.
"""

from __future__ import annotations

import torch
from torch import nn

from petinct.training.anatomy_controlnet import anatomy_to_8bit_channels
from petinct.training.distribution_gated_decoder import (
    DistributionGateStage,
    OrganProbabilityTokenEncoder,
)


class FullResolutionDilatedResidualBlock(nn.Module):
    """Expand, mix spatially at one dilation, and project without resizing."""

    def __init__(self, width: int, expansion: int, dilation: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(width, expansion, kernel_size=1, bias=False),
            nn.GroupNorm(min(8, expansion), expansion, eps=1e-6),
            nn.SiLU(),
            nn.Conv3d(
                expansion,
                expansion,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                groups=expansion,
                bias=False,
            ),
            nn.GroupNorm(min(8, expansion), expansion, eps=1e-6),
            nn.SiLU(),
            nn.Conv3d(expansion, width, kernel_size=1),
        )
        self.scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.scale.to(value.dtype) * self.block(value)


class FullResolutionDilatedConditionalFCN(nn.Module):
    """Deterministic PET generator with full-resolution conditional features."""

    def __init__(
        self,
        *,
        num_organ_ids: int = 223,
        token_channels: int = 16,
        width: int = 8,
        expansions: tuple[int, ...] = (16, 32, 64, 64, 32, 16),
        dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 1),
        channels_last_3d: bool = False,
    ) -> None:
        super().__init__()
        if len(expansions) != len(dilations):
            raise ValueError("expansions and dilations must have equal length")
        self.num_organ_ids = int(num_organ_ids)
        self.channels_last_3d = bool(channels_last_3d)
        self.anatomy_controlnet = nn.Sequential(
            nn.Conv3d(8, width, kernel_size=1, bias=False),
            nn.GroupNorm(min(8, width), width, eps=1e-6),
            nn.SiLU(),
        )
        self.ct_stem = nn.Sequential(
            nn.Conv3d(1, width, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(8, width), width, eps=1e-6),
            nn.SiLU(),
        )
        self.distribution_token_encoder = OrganProbabilityTokenEncoder(
            num_organ_ids=num_organ_ids,
            token_channels=token_channels,
        )
        self.distribution_decoder_gates = nn.ModuleDict(
            {"full": DistributionGateStage(token_channels, width)}
        )
        self.blocks = nn.ModuleList(
            [
                FullResolutionDilatedResidualBlock(width, expansion, dilation)
                for expansion, dilation in zip(expansions, dilations)
            ]
        )
        self.head = nn.Sequential(
            nn.GroupNorm(min(8, width), width, eps=1e-6),
            nn.SiLU(),
            nn.Conv3d(width, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.head[-2].bias, -2.5)

    def _paint_tokens(
        self,
        anatomy: torch.Tensor,
        organ_tokens: torch.Tensor,
    ) -> torch.Tensor:
        labels = anatomy[:, 0].long()
        if labels.min() < 0 or labels.max() >= organ_tokens.shape[1]:
            raise ValueError("anatomy label exceeds organ-token vocabulary")
        batch_index = torch.arange(
            labels.shape[0], device=labels.device
        ).view(-1, 1, 1, 1)
        dense = organ_tokens[batch_index, labels]
        memory_format = (
            torch.channels_last_3d
            if self.channels_last_3d
            else torch.contiguous_format
        )
        return dense.permute(0, 4, 1, 2, 3).contiguous(
            memory_format=memory_format
        )

    def gate_values(self) -> dict[str, torch.Tensor]:
        return {
            key: torch.tanh(stage.gate_logit)
            for key, stage in self.distribution_decoder_gates.items()
        }

    def forward(
        self,
        ct: torch.Tensor,
        anatomy: torch.Tensor,
        predicted_pet_probability: torch.Tensor,
        organ_mask: torch.Tensor,
        *,
        sample: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del sample
        anatomy_bits = anatomy_to_8bit_channels(anatomy).to(ct.dtype)
        if self.channels_last_3d:
            ct = ct.contiguous(memory_format=torch.channels_last_3d)
            anatomy_bits = anatomy_bits.contiguous(
                memory_format=torch.channels_last_3d
            )
        value = self.ct_stem(ct) + self.anatomy_controlnet(anatomy_bits)
        tokens = self.distribution_token_encoder(
            predicted_pet_probability, organ_mask
        )
        condition = self._paint_tokens(anatomy, tokens).to(ct.dtype)
        value = self.distribution_decoder_gates["full"](value, condition)
        for block in self.blocks:
            value = block(value)
        prediction = self.head(value)
        # Deterministic compatibility tensors make KL exactly zero.
        mu = prediction.new_zeros((prediction.shape[0], 1, 1, 1, 1))
        sigma = prediction.new_ones((prediction.shape[0], 1, 1, 1, 1))
        return prediction, mu, sigma
