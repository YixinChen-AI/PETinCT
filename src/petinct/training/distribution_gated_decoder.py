"""Whole-body PET-distribution gated decoder for AVAP.

The frozen WORM predictor supplies one 128-bin PET SUV probability
distribution for every present organ.  This module converts those
distributions into compact organ tokens, paints the tokens back into the
patch with the fused anatomy map, and adds them at three decoder resolutions
through zero-initialized learnable gates.  Consequently, a weights-only AVAP
warm start is functionally identical to the source model before optimization.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from petinct.training.anatomy_controlnet import (
    AnatomyControlledVariationalMaisiTranslator,
    anatomy_to_8bit_channels,
)


class OrganProbabilityTokenEncoder(nn.Module):
    """Encode an organ's SUV probability distribution and canonical identity."""

    def __init__(
        self,
        *,
        num_organ_ids: int = 223,
        token_channels: int = 32,
    ) -> None:
        super().__init__()
        if num_organ_ids < 2 or token_channels < 4:
            raise ValueError("invalid organ-token dimensions")
        self.num_organ_ids = int(num_organ_ids)
        self.token_channels = int(token_channels)
        self.quantile_encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, token_channels),
        )
        self.organ_embedding = nn.Embedding(
            num_organ_ids,
            token_channels,
            padding_idx=0,
        )
        self.output_norm = nn.LayerNorm(token_channels)

    def forward(
        self,
        predicted_pet_probability: torch.Tensor,
        organ_mask: torch.Tensor,
    ) -> torch.Tensor:
        if predicted_pet_probability.ndim != 3:
            raise ValueError(
                "predicted PET probability must be [B,organ,128]"
            )
        batch, organs, bins = predicted_pet_probability.shape
        if organs != self.num_organ_ids or bins != 128:
            raise ValueError(
                "predicted PET probability shape mismatch: "
                f"{tuple(predicted_pet_probability.shape)}"
            )
        if organ_mask.shape != (batch, organs) or organ_mask.dtype != torch.bool:
            raise ValueError("organ mask must be boolean [B,organ]")
        if organ_mask[:, 0].any():
            raise ValueError("background ID 0 cannot be a predicted organ")
        # The cache loader validates finite values, range and unit mass before
        # constructing this dense tensor.  Keep the neural path at the fixed
        # [B, 223, 128] shape: boolean compaction made every DDP rank compile
        # a different graph and left the other ranks waiting at synchronization.
        mask = organ_mask.unsqueeze(-1)
        probability = predicted_pet_probability.float().masked_fill(~mask, 0)
        # A uniform histogram is O(1) rather than O(1/128), which keeps the
        # probability branch numerically comparable to the ID branch.
        curve_token = self.quantile_encoder(
            (probability.reshape(batch * organs, 1, bins) * float(bins))
        ).reshape(batch, organs, self.token_channels)
        organ_ids = torch.arange(
            organs,
            device=predicted_pet_probability.device,
            dtype=torch.long,
        ).view(1, organs).expand(batch, -1)
        identity = self.organ_embedding(organ_ids)
        dense = self.output_norm(curve_token + identity)
        return dense * mask.to(dense.dtype)


# Historical import compatibility.  New code and provenance must use the
# probability name; the tensor contract itself is no longer quantile-based.
OrganQuantileTokenEncoder = OrganProbabilityTokenEncoder


class DistributionGateStage(nn.Module):
    """Project a condition map and expose it through a learnable zero gate."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1),
            nn.GroupNorm(min(32, out_channels), out_channels, eps=1e-6),
            nn.SiLU(),
            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                groups=out_channels,
            ),
        )
        self.gate_logit = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        decoded: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        if decoded.shape[2:] != condition.shape[2:]:
            condition = F.interpolate(
                condition,
                size=decoded.shape[2:],
                mode="trilinear",
                align_corners=False,
            )
        residual = self.projection(condition)
        if residual.shape != decoded.shape:
            raise RuntimeError(
                "decoder condition mismatch: "
                f"residual={tuple(residual.shape)} "
                f"decoded={tuple(decoded.shape)}"
            )
        return decoded + torch.tanh(self.gate_logit) * residual


class DistributionGatedAnatomyControlledTranslator(
    AnatomyControlledVariationalMaisiTranslator
):
    """AVAP with CT-only predicted organ distributions gated into its decoder."""

    _DECODER_STAGE_CHANNELS = {
        4: 256,
        2: 128,
        1: 64,
    }

    def __init__(
        self,
        autoencoder: nn.Module,
        *,
        num_organ_ids: int = 223,
        token_channels: int = 32,
    ) -> None:
        super().__init__(autoencoder)
        if not hasattr(self.decoder, "blocks"):
            raise ValueError("distribution gating requires MAISI decoder blocks")
        self.num_organ_ids = int(num_organ_ids)
        self.distribution_token_encoder = OrganProbabilityTokenEncoder(
            num_organ_ids=num_organ_ids,
            token_channels=token_channels,
        )
        self.distribution_decoder_gates = nn.ModuleDict(
            {
                str(scale): DistributionGateStage(
                    token_channels,
                    channels,
                )
                for scale, channels in self._DECODER_STAGE_CHANNELS.items()
            }
        )

    @staticmethod
    def _paint_tokens(
        anatomy: torch.Tensor,
        organ_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if anatomy.ndim != 5 or anatomy.shape[1] != 1:
            raise ValueError("anatomy must be [B,1,D,H,W]")
        labels = anatomy[:, 0].long()
        if labels.min() < 0 or labels.max() >= organ_tokens.shape[1]:
            raise ValueError("anatomy label exceeds organ-token vocabulary")
        batch_index = torch.arange(
            labels.shape[0],
            device=labels.device,
        ).view(-1, 1, 1, 1)
        dense = organ_tokens[batch_index, labels]
        return dense.permute(0, 4, 1, 2, 3).contiguous()

    def gate_values(self) -> dict[str, torch.Tensor]:
        return {
            scale: torch.tanh(stage.gate_logit)
            for scale, stage in self.distribution_decoder_gates.items()
        }

    def forward(
        self,
        ct: torch.Tensor,
        anatomy: torch.Tensor,
        predicted_pet_probability: torch.Tensor,
        organ_mask: torch.Tensor,
        *,
        sample: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        anatomy_channels = anatomy_to_8bit_channels(anatomy).to(ct.dtype)
        control_residuals = self.anatomy_controlnet(anatomy_channels)
        encoded = ct
        for block_index, block in enumerate(self.encoder.blocks):
            encoded = block(encoded)
            residual_index = self._INJECTION_BLOCKS.get(block_index)
            if residual_index is not None:
                residual = control_residuals[residual_index]
                if encoded.shape != residual.shape:
                    raise RuntimeError(
                        "anatomy residual shape mismatch at encoder block "
                        f"{block_index}"
                    )
                encoded = encoded + residual

        mu = self.to_mu(encoded).float()
        log_variance = self.to_log_sigma(encoded).float().clamp(-30.0, 20.0)
        sigma = torch.exp(0.5 * log_variance)
        latent = mu + sigma * torch.randn_like(sigma) if sample else mu

        tokens = self.distribution_token_encoder(
            predicted_pet_probability,
            organ_mask,
        )
        condition = self._paint_tokens(anatomy, tokens).to(ct.dtype)
        decoded = self.from_latent(latent)
        injected: set[str] = set()
        full_size = anatomy.shape[2:]
        for block in self.decoder.blocks:
            decoded = block(decoded)
            for scale, expected_channels in self._DECODER_STAGE_CHANNELS.items():
                key = str(scale)
                target_size = tuple(size // scale for size in full_size)
                if (
                    key not in injected
                    and decoded.shape[2:] == target_size
                    and decoded.shape[1] == expected_channels
                ):
                    decoded = self.distribution_decoder_gates[key](
                        decoded,
                        condition,
                    )
                    injected.add(key)
        expected = set(self.distribution_decoder_gates)
        if injected != expected:
            raise RuntimeError(
                f"decoder did not visit every gated scale: "
                f"injected={sorted(injected)} expected={sorted(expected)}"
            )
        return decoded, mu, sigma
