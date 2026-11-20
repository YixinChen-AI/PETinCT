"""Whole-body organ metabolic relation network (WOMR-Net)."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class SetAttentionBlock(nn.Module):
    """Permutation-equivariant self-attention block without positions."""

    def __init__(
        self,
        *,
        token_dim: int,
        num_heads: int,
        feedforward_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if token_dim % num_heads:
            raise ValueError("token_dim must be divisible by num_heads")
        self.attention_norm = nn.LayerNorm(token_dim)
        self.attention = nn.MultiheadAttention(
            token_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.feedforward_norm = nn.LayerNorm(token_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(token_dim, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, token_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        token: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        squeeze_batch = token.ndim == 2
        if squeeze_batch:
            token = token.unsqueeze(0)
        elif token.ndim != 3:
            raise ValueError(
                "set attention expects [tokens, channels] or "
                "[batch, tokens, channels]"
            )
        if key_padding_mask is not None:
            if (
                key_padding_mask.shape != token.shape[:2]
                or key_padding_mask.dtype is not torch.bool
            ):
                raise ValueError("invalid set-attention padding mask")
        elif not squeeze_batch:
            key_padding_mask = torch.zeros(
                token.shape[:2],
                dtype=torch.bool,
                device=token.device,
            )
        normalized = self.attention_norm(token)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        token = token + self.attention_dropout(attended)
        token = token + self.feedforward(self.feedforward_norm(token))
        if key_padding_mask is not None:
            token = token.masked_fill(key_padding_mask[..., None], 0)
        return token.squeeze(0) if squeeze_batch else token


class DistributionConvEncoder(nn.Module):
    """Encode an ordered 128-bin probability distribution with 1-D CNNs."""

    def __init__(self, output_dim: int = 128) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.Conv1d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.projection = nn.Linear(128, output_dim)

    def forward(self, probability: torch.Tensor) -> torch.Tensor:
        if probability.ndim != 2 or probability.shape[-1] != 128:
            raise ValueError(
                "probability must have shape [n_organs, 128], got "
                f"{tuple(probability.shape)}"
            )
        encoded = self.network(probability.unsqueeze(1)).squeeze(-1)
        return self.projection(encoded)


class SymmetricPairConditionedHead(nn.Module):
    """Shared scorer for arbitrary unordered organ pairs.

    The scorer uses commutative pair features, then explicitly symmetrizes its
    output so dropout cannot make score(i, j) differ from score(j, i).
    Rows are processed in chunks to bound peak activation memory.
    """

    def __init__(
        self,
        *,
        relation_dim: int,
        organ_id_dim: int,
        content_dim: int,
        identity_dim: int,
        hidden_dim: int,
        chunk_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if min(content_dim, identity_dim, hidden_dim, chunk_size) <= 0:
            raise ValueError("pair-head dimensions and chunk size must be positive")
        self.chunk_size = int(chunk_size)
        self.content_projection = nn.Linear(relation_dim, content_dim)
        self.identity_projection = nn.Linear(organ_id_dim, identity_dim)
        feature_dim = 3 * (content_dim + identity_dim)
        self.scorer = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    @staticmethod
    def _commutative_features(
        left: torch.Tensor,
        right: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat(
            [left + right, (left - right).abs(), left * right],
            dim=-1,
        )

    def forward(
        self,
        relation_embedding: torch.Tensor,
        organ_identity_embedding: torch.Tensor,
    ) -> torch.Tensor:
        if relation_embedding.ndim != 2:
            raise ValueError("relation_embedding must be [organs, channels]")
        if (
            organ_identity_embedding.ndim != 2
            or organ_identity_embedding.shape[0]
            != relation_embedding.shape[0]
        ):
            raise ValueError("organ identity embeddings do not match organs")
        content = self.content_projection(relation_embedding)
        identity = self.identity_projection(organ_identity_embedding)
        count = int(content.shape[0])
        rows: list[torch.Tensor] = []
        for start in range(0, count, self.chunk_size):
            stop = min(start + self.chunk_size, count)
            left_content = content[start:stop, None, :]
            right_content = content[None, :, :]
            left_identity = identity[start:stop, None, :]
            right_identity = identity[None, :, :]
            pair_feature = torch.cat(
                [
                    self._commutative_features(
                        left_content,
                        right_content,
                    ),
                    self._commutative_features(
                        left_identity,
                        right_identity,
                    ),
                ],
                dim=-1,
            )
            rows.append(self.scorer(pair_feature).squeeze(-1))
        raw = torch.cat(rows, dim=0)
        return 0.5 * (raw + raw.transpose(0, 1))


class WholeBodyOrganMetabolicRelationNet(nn.Module):
    """Predict a symmetric patient-consistency matrix between organ tokens.

    No volume, voxel-count, CT, image, or AVAE feature enters this model.
    Each organ is represented only by its 128-bin probability distribution
    and canonical organ ID.
    """

    def __init__(
        self,
        *,
        num_organ_ids: int,
        distribution_dim: int = 128,
        organ_id_dim: int = 64,
        token_dim: int = 256,
        relation_dim: int = 96,
        dropout: float = 0.1,
        set_attention_layers: int = 0,
        set_attention_heads: int = 8,
        set_attention_feedforward_dim: int = 1024,
        relation_head: str = "cosine",
        pair_content_dim: int = 24,
        pair_identity_dim: int = 12,
        pair_hidden_dim: int = 128,
        pair_chunk_size: int = 32,
    ) -> None:
        super().__init__()
        if num_organ_ids < 2:
            raise ValueError("num_organ_ids must include padding and foreground")
        if min(
            distribution_dim,
            organ_id_dim,
            token_dim,
            relation_dim,
        ) <= 0:
            raise ValueError("all embedding dimensions must be positive")
        self.num_organ_ids = int(num_organ_ids)
        self.set_attention_layers = int(set_attention_layers)
        if self.set_attention_layers < 0:
            raise ValueError("set_attention_layers cannot be negative")
        if relation_head not in {"cosine", "symmetric_pair"}:
            raise ValueError(
                "relation_head must be 'cosine' or 'symmetric_pair'"
            )
        self.relation_head = relation_head
        self.distribution_encoder = DistributionConvEncoder(distribution_dim)
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
        self.set_attention = nn.ModuleList(
            [
                SetAttentionBlock(
                    token_dim=token_dim,
                    num_heads=set_attention_heads,
                    feedforward_dim=set_attention_feedforward_dim,
                    dropout=dropout,
                )
                for _ in range(self.set_attention_layers)
            ]
        )
        self.relation_projection = nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_dim, relation_dim),
        )
        self.logit_scale = nn.Parameter(
            torch.tensor(math.log(10.0), dtype=torch.float32),
            requires_grad=relation_head == "cosine",
        )
        self.logit_bias = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.symmetric_pair_head = (
            SymmetricPairConditionedHead(
                relation_dim=relation_dim,
                organ_id_dim=organ_id_dim,
                content_dim=pair_content_dim,
                identity_dim=pair_identity_dim,
                hidden_dim=pair_hidden_dim,
                chunk_size=pair_chunk_size,
                dropout=dropout,
            )
            if relation_head == "symmetric_pair"
            else None
        )

    def encode_organ_tokens(
        self,
        probability: torch.Tensor,
        organ_ids: torch.Tensor,
        organ_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return valid organ embeddings in row-major batch order."""
        if probability.ndim != 3 or probability.shape[-1] != 128:
            raise ValueError(
                "probability must have shape [batch, organs, 128]"
            )
        if organ_ids.shape != probability.shape[:2]:
            raise ValueError("organ_ids shape does not match probability")
        if organ_mask.shape != probability.shape[:2]:
            raise ValueError("organ_mask shape does not match probability")
        if organ_mask.dtype is not torch.bool:
            raise TypeError("organ_mask must be boolean")
        valid_probability = probability[organ_mask]
        valid_ids = organ_ids[organ_mask]
        if valid_probability.shape[0] < 2:
            raise ValueError("at least two valid organs are required")
        if torch.any(valid_ids <= 0) or torch.any(
            valid_ids >= self.num_organ_ids
        ):
            raise ValueError("valid organ ID outside embedding range")
        distribution = self.distribution_encoder(valid_probability)
        identity = self.organ_id_embedding(valid_ids)
        return self.token_projection(torch.cat([distribution, identity], -1))

    def encode_organs(
        self,
        probability: torch.Tensor,
        organ_ids: torch.Tensor,
        organ_mask: torch.Tensor,
    ) -> torch.Tensor:
        token = self.encode_organ_tokens(
            probability,
            organ_ids,
            organ_mask,
        )
        for block in self.set_attention:
            token = block(token)
        return F.normalize(self.relation_projection(token), dim=-1)

    def score_relation_pairs(
        self,
        relation_embedding: torch.Tensor,
        organ_ids: torch.Tensor,
    ) -> torch.Tensor:
        if organ_ids.shape != (relation_embedding.shape[0],):
            raise ValueError("organ_ids do not match relation embeddings")
        if self.symmetric_pair_head is not None:
            identity = self.organ_id_embedding(organ_ids)
            return (
                self.symmetric_pair_head(
                    relation_embedding,
                    identity,
                ).float()
                + self.logit_bias.float()
            )
        scale = self.logit_scale.exp().clamp(max=100.0)
        return (
            scale
            * relation_embedding.float()
            @ relation_embedding.float().transpose(0, 1)
            + self.logit_bias.float()
        )

    def forward(
        self,
        probability: torch.Tensor,
        organ_ids: torch.Tensor,
        organ_mask: torch.Tensor,
        case_episode_index: torch.Tensor | None = None,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]
    ):
        """Return one full matrix or independent episode matrices."""
        if case_episode_index is not None:
            return self.relation_blocks(
                probability,
                organ_ids,
                organ_mask,
                case_episode_index,
            )
        relation_embedding = self.encode_organs(
            probability,
            organ_ids,
            organ_mask,
        )
        logits = self.score_relation_pairs(
            relation_embedding,
            organ_ids[organ_mask],
        )
        return logits, relation_embedding

    def relation_blocks(
        self,
        probability: torch.Tensor,
        organ_ids: torch.Tensor,
        organ_mask: torch.Tensor,
        case_episode_index: torch.Tensor,
    ) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
        """Return one independent relation matrix per patient episode."""
        if case_episode_index.shape != (probability.shape[0],):
            raise ValueError("case_episode_index does not match case batch")
        organ_token = self.encode_organ_tokens(
            probability,
            organ_ids,
            organ_mask,
        )
        valid_episode = case_episode_index[:, None].expand_as(organ_mask)[
            organ_mask
        ]
        valid_ids = organ_ids[organ_mask]
        blocks: list[torch.Tensor] = []
        contextual_embeddings: list[torch.Tensor] = []
        for episode in torch.unique(valid_episode, sorted=True):
            episode_mask = valid_episode == episode
            selected = organ_token[episode_mask]
            for attention_block in self.set_attention:
                selected = attention_block(selected)
            relation_embedding = F.normalize(
                self.relation_projection(selected),
                dim=-1,
            )
            contextual_embeddings.append(relation_embedding)
            blocks.append(
                self.score_relation_pairs(
                    relation_embedding,
                    valid_ids[episode_mask],
                )
            )
        return blocks, torch.cat(contextual_embeddings, dim=0), valid_episode
