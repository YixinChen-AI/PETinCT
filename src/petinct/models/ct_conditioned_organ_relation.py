"""CT-conditioned whole-body organ metabolic relation network."""

from __future__ import annotations

import torch
from torch import nn

from petinct.models.organ_relation import (
    DistributionConvEncoder,
    SetAttentionBlock,
    WholeBodyOrganMetabolicRelationNet,
)


class CtPetCompatibilityHead(nn.Module):
    """Score a candidate PET organ against one whole-body CT patient anchor."""

    def __init__(
        self,
        *,
        relation_dim: int,
        token_dim: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.pet_projection = nn.Linear(relation_dim, hidden_dim)
        self.ct_projection = nn.Linear(token_dim, hidden_dim)
        self.global_projection = nn.Linear(token_dim, hidden_dim)
        self.scorer = nn.Sequential(
            nn.LayerNorm(hidden_dim * 5),
            nn.Linear(hidden_dim * 5, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        pet: torch.Tensor,
        ct: torch.Tensor,
        global_ct: torch.Tensor,
    ) -> torch.Tensor:
        pet = self.pet_projection(pet)
        ct = self.ct_projection(ct)
        global_ct = self.global_projection(global_ct)
        feature = torch.cat(
            [pet, ct, global_ct, (pet - ct).abs(), pet * ct],
            dim=-1,
        )
        return self.scorer(feature).squeeze(-1)


class CtConditionedWOMRNet(nn.Module):
    """Joint PET-PET relation and CT-anchored PET compatibility model."""

    def __init__(
        self,
        *,
        num_organ_ids: int,
        distribution_dim: int = 128,
        organ_id_dim: int = 64,
        token_dim: int = 256,
        relation_dim: int = 96,
        set_attention_layers: int = 4,
        set_attention_heads: int = 8,
        set_attention_feedforward_dim: int = 1024,
        pair_content_dim: int = 24,
        pair_identity_dim: int = 12,
        pair_hidden_dim: int = 128,
        pair_chunk_size: int = 32,
        compatibility_hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_organ_ids = int(num_organ_ids)
        self.pet_relation = WholeBodyOrganMetabolicRelationNet(
            num_organ_ids=num_organ_ids,
            distribution_dim=distribution_dim,
            organ_id_dim=organ_id_dim,
            token_dim=token_dim,
            relation_dim=relation_dim,
            dropout=dropout,
            set_attention_layers=set_attention_layers,
            set_attention_heads=set_attention_heads,
            set_attention_feedforward_dim=set_attention_feedforward_dim,
            relation_head="symmetric_pair",
            pair_content_dim=pair_content_dim,
            pair_identity_dim=pair_identity_dim,
            pair_hidden_dim=pair_hidden_dim,
            pair_chunk_size=pair_chunk_size,
        )
        self.ct_distribution_encoder = DistributionConvEncoder(
            distribution_dim
        )
        self.ct_token_projection = nn.Sequential(
            nn.Linear(distribution_dim + organ_id_dim, token_dim),
            nn.LayerNorm(token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_dim, token_dim),
            nn.LayerNorm(token_dim),
        )
        self.ct_set_attention = nn.ModuleList(
            [
                SetAttentionBlock(
                    token_dim=token_dim,
                    num_heads=set_attention_heads,
                    feedforward_dim=set_attention_feedforward_dim,
                    dropout=dropout,
                )
                for _ in range(set_attention_layers)
            ]
        )
        self.compatibility_head = CtPetCompatibilityHead(
            relation_dim=relation_dim,
            token_dim=token_dim,
            hidden_dim=compatibility_hidden_dim,
            dropout=dropout,
        )

    def _encode_ct_cases(
        self,
        probability: torch.Tensor,
        organ_ids: torch.Tensor,
        organ_mask: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        encoded_cases: list[torch.Tensor] = []
        global_cases: list[torch.Tensor] = []
        for case_index in range(probability.shape[0]):
            selected = organ_mask[case_index]
            ids = organ_ids[case_index, selected]
            if ids.numel() < 1:
                raise ValueError("CT patient anchor has no valid organs")
            distribution = self.ct_distribution_encoder(
                probability[case_index, selected]
            )
            identity = self.pet_relation.organ_id_embedding(ids)
            token = self.ct_token_projection(
                torch.cat([distribution, identity], dim=-1)
            )
            for block in self.ct_set_attention:
                token = block(token)
            encoded_cases.append(token)
            global_cases.append(token.mean(dim=0))
        return encoded_cases, global_cases

    def _compatibility_blocks(
        self,
        *,
        pet_embedding: torch.Tensor,
        organ_ids: torch.Tensor,
        organ_mask: torch.Tensor,
        patient_index: torch.Tensor,
        case_episode_index: torch.Tensor,
        ct_cases: list[torch.Tensor],
        global_ct: list[torch.Tensor],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        valid_ids = organ_ids[organ_mask]
        valid_patient = patient_index[:, None].expand_as(organ_mask)[
            organ_mask
        ]
        valid_case = (
            torch.arange(
                organ_mask.shape[0],
                device=organ_mask.device,
            )[:, None]
            .expand_as(organ_mask)[organ_mask]
        )
        valid_episode = case_episode_index[:, None].expand_as(organ_mask)[
            organ_mask
        ]
        logits: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        for episode in torch.unique(case_episode_index, sorted=True):
            episode_cases = torch.nonzero(
                case_episode_index == episode,
                as_tuple=False,
            ).flatten()
            candidate_mask = valid_episode == episode
            candidate_pet = pet_embedding[candidate_mask]
            candidate_ids = valid_ids[candidate_mask]
            candidate_patient = valid_patient[candidate_mask]
            candidate_case = valid_case[candidate_mask]
            if not torch.equal(
                candidate_case.unique(sorted=True),
                episode_cases,
            ):
                raise RuntimeError("PET episode/case ownership mismatch")
            for anchor_case_tensor in episode_cases:
                anchor_case = int(anchor_case_tensor.item())
                anchor_ids = organ_ids[anchor_case, organ_mask[anchor_case]]
                anchor_ct = ct_cases[anchor_case]
                if anchor_ids.shape[0] != anchor_ct.shape[0]:
                    raise RuntimeError("CT organ/token count mismatch")
                anchor_patient = patient_index[anchor_case]
                for ct_row, organ_id in enumerate(anchor_ids):
                    selected = candidate_ids == organ_id
                    if not selected.any():
                        continue
                    pet_selected = candidate_pet[selected]
                    ct_selected = anchor_ct[ct_row].expand(
                        pet_selected.shape[0], -1
                    )
                    global_selected = global_ct[anchor_case].expand(
                        pet_selected.shape[0], -1
                    )
                    logits.append(
                        self.compatibility_head(
                            pet_selected,
                            ct_selected,
                            global_selected,
                        )
                    )
                    targets.append(
                        (candidate_patient[selected] == anchor_patient).to(
                            pet_selected.dtype
                        )
                    )
        if not logits:
            raise RuntimeError("no same-organ CT/PET compatibility pairs")
        return logits, targets

    def forward(
        self,
        pet_probability: torch.Tensor,
        ct_probability: torch.Tensor,
        organ_ids: torch.Tensor,
        organ_mask: torch.Tensor,
        patient_index: torch.Tensor,
        case_episode_index: torch.Tensor,
    ) -> dict[str, object]:
        if pet_probability.shape != ct_probability.shape:
            raise ValueError("CT/PET probability tensor shapes differ")
        relation_blocks, pet_embedding, valid_episode = (
            self.pet_relation.relation_blocks(
                pet_probability,
                organ_ids,
                organ_mask,
                case_episode_index,
            )
        )
        expected_episode = case_episode_index[:, None].expand_as(organ_mask)[
            organ_mask
        ]
        if not torch.equal(valid_episode, expected_episode):
            raise RuntimeError("PET relation embedding order mismatch")
        ct_cases, global_ct = self._encode_ct_cases(
            ct_probability,
            organ_ids,
            organ_mask,
        )
        compatibility_logits, compatibility_targets = (
            self._compatibility_blocks(
                pet_embedding=pet_embedding,
                organ_ids=organ_ids,
                organ_mask=organ_mask,
                patient_index=patient_index,
                case_episode_index=case_episode_index,
                ct_cases=ct_cases,
                global_ct=global_ct,
            )
        )
        return {
            "relation_blocks": relation_blocks,
            "pet_embedding": pet_embedding,
            "valid_episode": valid_episode,
            "compatibility_logits": compatibility_logits,
            "compatibility_targets": compatibility_targets,
        }
