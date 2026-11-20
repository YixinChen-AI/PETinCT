"""Patient-level dataset for the WOMR-Net organ relation model."""

from __future__ import annotations

import csv
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


REQUIRED_CASE_COLUMNS = {
    "case_id",
    "patient_group_hash",
    "split",
    "distribution_path",
}


def cdf_to_probability(cdf: np.ndarray) -> np.ndarray:
    """Recover an empirical PDF from a valid stored CDF."""
    values = np.asarray(cdf, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 128:
        raise ValueError(f"cdf must have shape [n_organs,128], got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("CDF contains non-finite values")
    if np.any(np.diff(values, axis=1) < -1e-5):
        raise ValueError("CDF is not monotonic")
    if not np.allclose(values[:, -1], 1.0, atol=1e-5, rtol=0.0):
        raise ValueError("CDF terminal mass is not one")
    probability = np.diff(
        values,
        axis=1,
        prepend=np.zeros((values.shape[0], 1), dtype=np.float32),
    )
    np.maximum(probability, 0.0, out=probability)
    mass = probability.sum(axis=1, keepdims=True)
    if np.any(~np.isfinite(mass)) or np.any(mass <= 0):
        raise ValueError("recovered PDF has invalid mass")
    probability /= mass
    return probability.astype(np.float32, copy=False)


class OrganDistributionCaseDataset(Dataset[dict[str, Any]]):
    """Load every present organ distribution for one case."""

    def __init__(
        self,
        *,
        library_root: str | Path,
        split: str,
    ) -> None:
        self.root = Path(library_root)
        manifest = self.root / "case_manifest.csv"
        with manifest.open(newline="") as handle:
            reader = csv.DictReader(handle)
            missing = REQUIRED_CASE_COLUMNS.difference(
                reader.fieldnames or ()
            )
            if missing:
                raise ValueError(
                    f"case manifest missing columns {sorted(missing)}"
                )
            self.rows = [row for row in reader if row["split"] == split]
        if not self.rows:
            raise ValueError(f"no cases found for split={split!r}")
        case_ids = [row["case_id"] for row in self.rows]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError(f"duplicate case IDs in split={split!r}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        path = self.root / row["distribution_path"]
        with np.load(path, allow_pickle=False) as archive:
            required = {
                "case_id",
                "patient_group_hash",
                "model_label_ids",
                "cdf_128",
            }
            missing = required.difference(archive.files)
            if missing:
                raise ValueError(
                    f"distribution archive missing {sorted(missing)}: {path}"
                )
            case_id = str(archive["case_id"].item())
            patient = str(archive["patient_group_hash"].item())
            organ_ids = np.asarray(
                archive["model_label_ids"], dtype=np.int64
            )
            probability = cdf_to_probability(archive["cdf_128"])
        if case_id != row["case_id"] or patient != row["patient_group_hash"]:
            raise ValueError(f"archive/manifest identity mismatch: {path}")
        if (
            organ_ids.ndim != 1
            or organ_ids.shape[0] != probability.shape[0]
            or np.any(organ_ids <= 0)
            or len(set(int(value) for value in organ_ids)) != len(organ_ids)
        ):
            raise ValueError(f"invalid organ IDs in {path}")
        return {
            "case_id": case_id,
            "patient_group_hash": patient,
            "organ_ids": torch.from_numpy(organ_ids),
            "probability": torch.from_numpy(probability),
        }


def collate_organ_cases(
    examples: list[dict[str, Any]],
) -> dict[str, Any]:
    """Pad variable organ counts while retaining an explicit validity mask."""
    if not examples:
        raise ValueError("cannot collate an empty batch")
    batch_size = len(examples)
    maximum = max(int(item["organ_ids"].shape[0]) for item in examples)
    probability = torch.zeros(
        (batch_size, maximum, 128),
        dtype=torch.float32,
    )
    organ_ids = torch.zeros((batch_size, maximum), dtype=torch.long)
    organ_mask = torch.zeros((batch_size, maximum), dtype=torch.bool)
    patient_lookup: dict[str, int] = {}
    patient_index = torch.empty(batch_size, dtype=torch.long)
    for batch_index, item in enumerate(examples):
        count = int(item["organ_ids"].shape[0])
        probability[batch_index, :count] = item["probability"]
        organ_ids[batch_index, :count] = item["organ_ids"]
        organ_mask[batch_index, :count] = True
        patient = item["patient_group_hash"]
        patient_index[batch_index] = patient_lookup.setdefault(
            patient, len(patient_lookup)
        )
    return {
        "case_ids": [item["case_id"] for item in examples],
        "patient_group_hashes": [
            item["patient_group_hash"] for item in examples
        ],
        "patient_index": patient_index,
        "probability": probability,
        "organ_ids": organ_ids,
        "organ_mask": organ_mask,
    }


def collate_organ_episodes(
    examples: list[dict[str, Any]],
    *,
    patients_per_episode: int,
) -> dict[str, Any]:
    """Collate consecutive unique-patient groups as independent episodes."""
    if patients_per_episode < 2:
        raise ValueError("patients_per_episode must be at least two")
    batch = collate_organ_cases(examples)
    hashes = batch["patient_group_hashes"]
    episode_index = torch.arange(len(examples), dtype=torch.long)
    episode_index //= patients_per_episode
    for episode in torch.unique(episode_index):
        selected = [
            hashes[index]
            for index in torch.nonzero(
                episode_index == episode, as_tuple=False
            ).flatten().tolist()
        ]
        if len(selected) < 2:
            raise ValueError("every relation episode needs at least two patients")
        if len(selected) != len(set(selected)):
            raise ValueError("a patient is duplicated within one episode")
    batch["case_episode_index"] = episode_index
    batch["patients_per_episode"] = patients_per_episode
    batch["episodes_in_batch"] = int(episode_index.max().item()) + 1
    return batch


class PatientEpisodeBatchSampler(Sampler[list[int]]):
    """Yield batches containing independent groups of unique patients."""

    def __init__(
        self,
        dataset: OrganDistributionCaseDataset,
        *,
        patients_per_episode: int,
        episodes_per_batch: int,
        seed: int,
        shuffle: bool,
        drop_last: bool,
        num_replicas: int = 1,
        rank: int = 0,
    ) -> None:
        if patients_per_episode < 2 or episodes_per_batch < 1:
            raise ValueError("invalid episode dimensions")
        if num_replicas < 1 or not 0 <= rank < num_replicas:
            raise ValueError("invalid distributed sampler rank")
        self.dataset = dataset
        self.patients_per_episode = patients_per_episode
        self.episodes_per_batch = episodes_per_batch
        self.batch_patients = patients_per_episode * episodes_per_batch
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.epoch = 0
        self.patient_to_indices: dict[str, list[int]] = {}
        for index, row in enumerate(dataset.rows):
            self.patient_to_indices.setdefault(
                row["patient_group_hash"], []
            ).append(index)
        self.patients = sorted(self.patient_to_indices)
        if len(self.patients) < self.batch_patients:
            raise ValueError(
                f"only {len(self.patients)} patients for "
                f"batch requirement {self.batch_patients}"
            )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _rank_patients(self) -> list[str]:
        patients = list(self.patients)
        generator = random.Random(self.seed + self.epoch)
        if self.shuffle:
            generator.shuffle(patients)
        usable = len(patients) - len(patients) % self.num_replicas
        patients = patients[:usable][self.rank :: self.num_replicas]
        return patients

    def __iter__(self):
        patients = self._rank_patients()
        generator = random.Random(
            self.seed + 1_000_003 * self.epoch + self.rank
        )
        indices = [
            generator.choice(self.patient_to_indices[patient])
            for patient in patients
        ]
        full = len(indices) // self.batch_patients
        for batch_index in range(full):
            start = batch_index * self.batch_patients
            yield indices[start : start + self.batch_patients]
        remainder = indices[full * self.batch_patients :]
        if not self.drop_last:
            # A final one-patient episode has no cross-patient negatives.
            # Drop only that trailing identity while retaining all complete
            # episodes and any final episode with at least two patients.
            if len(remainder) % self.patients_per_episode == 1:
                remainder = remainder[:-1]
            if len(remainder) >= 2:
                yield remainder

    def __len__(self) -> int:
        per_rank = len(self.patients) // self.num_replicas
        if self.drop_last:
            return per_rank // self.batch_patients
        full, remainder = divmod(per_rank, self.batch_patients)
        if remainder % self.patients_per_episode == 1:
            remainder -= 1
        return full + int(remainder >= 2)
