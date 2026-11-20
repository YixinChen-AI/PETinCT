"""CT-conditioned whole-body organ PET distribution predictor."""

from __future__ import annotations

import torch
from torch import nn

from petinct.models.organ_relation import (
    DistributionConvEncoder,
    SetAttentionBlock,
)


class CtToPetOrganDistributionNet(nn.Module):
    """Predict every present organ's PET distribution from CT only.

    The organ axis is a set: no positional encoding is used, and canonical
    organ identity is represented only by the learned organ-ID embedding.
    """

    def __init__(
        self,
        *,
        num_organ_ids: int,
        distribution_dim: int = 128,
        output_distribution_dim: int = 128,
        organ_id_dim: int = 64,
        token_dim: int = 256,
        attention_layers: int = 4,
        attention_heads: int = 8,
        feedforward_dim: int = 1024,
        dropout: float = 0.1,
        context_mode: str = "full",
        roi_feature_dim: int | None = None,
    ) -> None:
        super().__init__()
        if num_organ_ids < 2:
            raise ValueError("num_organ_ids must include padding and foreground")
        if attention_layers < 1 or token_dim % attention_heads:
            raise ValueError("invalid Set Transformer dimensions")
        if context_mode not in {"full", "local", "organ_id_only"}:
            raise ValueError(f"unknown context mode: {context_mode}")
        self.num_organ_ids = int(num_organ_ids)
        self.output_distribution_dim = int(output_distribution_dim)
        if self.output_distribution_dim < 2:
            raise ValueError("output_distribution_dim must be at least two")
        self.context_mode = context_mode
        self.roi_feature_dim = (
            None if roi_feature_dim is None else int(roi_feature_dim)
        )
        self.ct_distribution_encoder = DistributionConvEncoder(
            distribution_dim
        )
        self.organ_id_embedding = nn.Embedding(
            num_organ_ids,
            organ_id_dim,
            padding_idx=0,
        )
        self.token_projection = nn.Sequential(
            nn.Linear(distribution_dim + organ_id_dim, token_dim),
            nn.LayerNorm(token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_dim, token_dim),
            nn.LayerNorm(token_dim),
        )
        if self.roi_feature_dim is not None:
            if self.roi_feature_dim < 1:
                raise ValueError("roi_feature_dim must be positive")
            self.roi_projection = nn.Sequential(
                nn.LayerNorm(self.roi_feature_dim),
                nn.Linear(self.roi_feature_dim, token_dim),
            )
            # A legacy WORM checkpoint is functionally unchanged until the
            # shared-encoder ROI branch proves useful during optimization.
            self.roi_gate_logit = nn.Parameter(torch.zeros(()))
        else:
            self.roi_projection = None
            self.register_parameter("roi_gate_logit", None)
        self.set_attention = nn.ModuleList(
            [
                SetAttentionBlock(
                    token_dim=token_dim,
                    num_heads=attention_heads,
                    feedforward_dim=feedforward_dim,
                    dropout=dropout,
                )
                for _ in range(attention_layers)
            ]
        )
        self.pet_distribution_decoder = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_dim, self.output_distribution_dim),
        )

    @staticmethod
    def _restore(
        valid: torch.Tensor,
        organ_mask: torch.Tensor,
    ) -> torch.Tensor:
        dense = valid.new_zeros((*organ_mask.shape, valid.shape[-1]))
        dense[organ_mask] = valid
        return dense

    def forward(
        self,
        ct_probability: torch.Tensor,
        organ_ids: torch.Tensor,
        organ_mask: torch.Tensor,
        *,
        ct_roi_features: torch.Tensor | None = None,
        output_mode: str = "probability",
    ) -> torch.Tensor:
        if ct_probability.ndim != 3 or ct_probability.shape[-1] != 128:
            raise ValueError("ct_probability must be [batch, organs, 128]")
        if organ_ids.shape != ct_probability.shape[:2]:
            raise ValueError("organ IDs do not align with CT distributions")
        if (
            organ_mask.shape != organ_ids.shape
            or organ_mask.dtype is not torch.bool
        ):
            raise ValueError("invalid organ mask")
        if torch.any(organ_ids[organ_mask] <= 0):
            raise ValueError("foreground organs require positive IDs")
        if torch.any(organ_ids[organ_mask] >= self.num_organ_ids):
            raise ValueError("organ ID exceeds embedding vocabulary")
        if torch.any(organ_mask.sum(dim=1) == 0):
            raise ValueError("every patient must contain an organ")

        valid_ids = organ_ids[organ_mask]
        if self.context_mode == "organ_id_only":
            # A fixed valid distribution removes every patient-specific CT
            # value while keeping the encoder and downstream capacity intact.
            ct_probability = torch.full_like(
                ct_probability,
                1.0 / ct_probability.shape[-1],
            )
        distribution = self.ct_distribution_encoder(
            ct_probability[organ_mask]
        )
        identity = self.organ_id_embedding(valid_ids)
        valid_token = self.token_projection(
            torch.cat([distribution, identity], dim=-1)
        )
        if ct_roi_features is not None:
            if self.roi_projection is None or self.roi_gate_logit is None:
                raise ValueError(
                    "ct_roi_features require a configured roi_feature_dim"
                )
            expected = (*organ_mask.shape, self.roi_feature_dim)
            if ct_roi_features.shape != expected:
                raise ValueError(
                    "CT ROI feature shape mismatch: "
                    f"{tuple(ct_roi_features.shape)} != {expected}"
                )
            # The shared AVAP encoder may emit bf16 features even when WORM
            # inference itself runs outside autocast.  Align the interface to
            # WORM's token path so LayerNorm/Linear never receive mixed
            # activation and parameter dtypes.
            valid_roi = ct_roi_features[organ_mask].to(
                device=valid_token.device,
                dtype=valid_token.dtype,
            )
            if not torch.isfinite(valid_roi).all():
                raise ValueError("CT ROI features contain non-finite values")
            valid_token = valid_token + torch.tanh(
                self.roi_gate_logit
            ) * self.roi_projection(valid_roi)
        token = self._restore(valid_token, organ_mask)
        padding_mask = ~organ_mask
        if self.context_mode == "local":
            # Process each organ as a length-one set. This retains the exact
            # Transformer parameter budget but forbids cross-organ messages.
            valid_token = token[organ_mask].unsqueeze(1)
            for block in self.set_attention:
                valid_token = block(valid_token, key_padding_mask=None)
            token = self._restore(valid_token[:, 0], organ_mask)
        else:
            for block in self.set_attention:
                token = block(token, key_padding_mask=padding_mask)
        logits = self.pet_distribution_decoder(token)
        if output_mode == "probability":
            output = torch.softmax(logits.float(), dim=-1)
        elif output_mode == "quantile":
            # Direct quantile prediction avoids a discontinuous inverse-CDF
            # operation in the optimization path. Sorting is differentiable
            # almost everywhere and enforces a valid monotone quantile curve.
            output = torch.sort(
                20.0 * torch.sigmoid(logits.float()),
                dim=-1,
            ).values
        else:
            raise ValueError(f"unknown output mode: {output_mode}")
        return output.masked_fill(padding_mask[..., None], 0.0)
