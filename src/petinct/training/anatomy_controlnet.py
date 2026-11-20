"""Anatomy ControlNet for joint CT-to-PET AVAE training.

The MAISI AVAE remains structurally unchanged.  A separate anatomy branch
produces zero-initialized residuals at three encoder resolutions.  At
initialization every residual is exactly zero, so a weights-only warm start is
functionally identical to its source checkpoint.  Whether the AVAE is frozen
or jointly optimized is an explicit training-entry decision; this module does
not freeze it implicitly.
"""

from __future__ import annotations

import torch
from torch import nn


def anatomy_to_8bit_channels(anatomy: torch.Tensor) -> torch.Tensor:
    """Encode fused integer anatomy labels as eight fixed binary channels."""

    if anatomy.ndim != 5 or anatomy.shape[1] != 1:
        raise ValueError(
            "anatomy must have shape [B,1,D,H,W], got "
            f"{tuple(anatomy.shape)}"
        )
    shifts = torch.arange(
        8, device=anatomy.device, dtype=torch.long
    ).view(1, 8, 1, 1, 1)
    return ((anatomy.long() >> shifts) & 1).to(torch.float32)


class _ControlBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
            ),
            nn.GroupNorm(min(32, out_channels), out_channels, eps=1e-6),
            nn.SiLU(),
            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                groups=out_channels,
            ),
            nn.GroupNorm(min(32, out_channels), out_channels, eps=1e-6),
            nn.SiLU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


def _zero_conv(channels: int) -> nn.Conv3d:
    layer = nn.Conv3d(channels, channels, kernel_size=1)
    nn.init.zeros_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer


class AnatomyControlNet3D(nn.Module):
    """Generate encoder residuals at 64^3, 32^3, and bottleneck resolution."""

    def __init__(self) -> None:
        super().__init__()
        self.stage_64 = _ControlBlock(8, 64, stride=2)
        self.stage_32 = _ControlBlock(64, 128, stride=2)
        self.stage_bottleneck = _ControlBlock(128, 256, stride=1)
        self.zero_64 = _zero_conv(64)
        self.zero_32 = _zero_conv(128)
        self.zero_bottleneck = _zero_conv(256)

    def forward(
        self, anatomy_channels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feature_64 = self.stage_64(anatomy_channels)
        feature_32 = self.stage_32(feature_64)
        feature_bottleneck = self.stage_bottleneck(feature_32)
        return (
            self.zero_64(feature_64),
            self.zero_32(feature_32),
            self.zero_bottleneck(feature_bottleneck),
        )


class AnatomyControlledVariationalMaisiTranslator(nn.Module):
    """Variational MAISI translator with multi-scale anatomy residuals."""

    # Full-depth MAISI encoder:
    # 3 -> 64 channels at 64^3, 6 -> 128 channels at 32^3,
    # 8 -> 256 channels at 32^3 before norm/output projection to 4 channels.
    _INJECTION_BLOCKS = {3: 0, 6: 1, 8: 2}

    def __init__(self, autoencoder: nn.Module) -> None:
        super().__init__()
        self.encoder = autoencoder.encoder
        self.to_mu = autoencoder.quant_conv_mu
        self.to_log_sigma = autoencoder.quant_conv_log_sigma
        self.from_latent = autoencoder.post_quant_conv
        self.decoder = autoencoder.decoder
        self.anatomy_controlnet = AnatomyControlNet3D()

        if len(self.encoder.blocks) != 11:
            raise ValueError(
                "anatomy ControlNet is pinned to the pretrained full-depth "
                f"MAISI encoder with 11 blocks, got {len(self.encoder.blocks)}"
            )

    def freeze_pretrained_backbone(self) -> None:
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name.startswith("anatomy_controlnet."))

    def encode_ct(
        self,
        ct: torch.Tensor,
        anatomy: torch.Tensor,
        *,
        sample: bool = True,
        feature_block_index: int | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
    ]:
        """Encode CT once and optionally expose one deterministic feature map.

        The returned feature tap is taken before posterior sampling, so the
        same patient produces stable organ ROI evidence.  It is intended for
        anatomy-indexed pooling, not for a second image encoder.
        """

        if feature_block_index is not None and not (
            0 <= feature_block_index < len(self.encoder.blocks)
        ):
            raise ValueError(
                "feature_block_index outside MAISI encoder block range"
            )
        anatomy_channels = anatomy_to_8bit_channels(anatomy).to(ct.dtype)
        control_residuals = self.anatomy_controlnet(anatomy_channels)

        encoded = ct
        feature_tap = None
        for block_index, block in enumerate(self.encoder.blocks):
            encoded = block(encoded)
            residual_index = self._INJECTION_BLOCKS.get(block_index)
            if residual_index is not None:
                residual = control_residuals[residual_index]
                if encoded.shape != residual.shape:
                    raise RuntimeError(
                        "ControlNet residual shape mismatch at encoder block "
                        f"{block_index}: encoded={tuple(encoded.shape)} "
                        f"residual={tuple(residual.shape)}"
                    )
                encoded = encoded + residual
            if block_index == feature_block_index:
                feature_tap = encoded

        mu = self.to_mu(encoded).float()
        log_variance = self.to_log_sigma(encoded).float().clamp(-30.0, 20.0)
        sigma = torch.exp(0.5 * log_variance)
        latent = mu + sigma * torch.randn_like(sigma) if sample else mu
        return latent, mu, sigma, feature_tap

    def forward(
        self,
        ct: torch.Tensor,
        anatomy: torch.Tensor,
        *,
        sample: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent, mu, sigma, _ = self.encode_ct(
            ct,
            anatomy,
            sample=sample,
        )
        prediction = self.decoder(self.from_latent(latent))
        return prediction, mu, sigma
