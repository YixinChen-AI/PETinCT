"""Generate resumable full-volume coarse PET for recurrent refinement.

This is an inference/cache builder, not an evaluator.  It selects a frozen
subset of the training split, reconstructs deterministic posterior-mean PET
with overlapping 128-cube sliding windows, and writes one geometry-preserving
SUV NIfTI plus one atomic provenance record per case.
"""

from __future__ import annotations

import argparse
import csv
import errno
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch
from monai.inferers import sliding_window_inference

from petinct.data.ct_norm import ct_normalize
from petinct.training.anatomy_controlnet import (
    AnatomyControlledVariationalMaisiTranslator,
)
from petinct.training.distribution_gated_decoder import (
    DistributionGatedAnatomyControlledTranslator,
)
from petinct.training.full_resolution_dilated_fcn import (
    FullResolutionDilatedConditionalFCN,
)
from petinct.training.vae import build_autoencoder


class VariationalMaisiTranslator(torch.nn.Module):
    """Inference-only posterior-mean wrapper without training dependencies."""

    def __init__(self, autoencoder: torch.nn.Module) -> None:
        super().__init__()
        self.encoder = autoencoder.encoder
        self.to_mu = autoencoder.quant_conv_mu
        self.to_log_sigma = autoencoder.quant_conv_log_sigma
        self.from_latent = autoencoder.post_quant_conv
        self.decoder = autoencoder.decoder

    def forward(
        self,
        ct: torch.Tensor,
        *,
        sample: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded = self.encoder(ct)
        mu = self.to_mu(encoded).float()
        log_variance = self.to_log_sigma(encoded).float().clamp(-30.0, 20.0)
        sigma = torch.exp(0.5 * log_variance)
        latent = mu + sigma * torch.randn_like(sigma) if sample else mu
        prediction = self.decoder(self.from_latent(latent))
        return prediction, mu, sigma


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_triplet(value: str) -> tuple[int, int, int]:
    parts = tuple(int(part) for part in value.replace("x", ",").split(","))
    if len(parts) != 3 or any(part < 1 for part in parts):
        raise ValueError(f"expected positive D,H,W triplet, got {value!r}")
    return parts


def load_rows(
    path: Path,
    splits: tuple[str, ...],
    seed: int,
    limit: int,
    num_shards: int,
    shard_index: int,
) -> tuple[list[dict[str, str]], int]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [
            row for row in csv.DictReader(handle) if row["split"] in splits
        ]
    if not rows:
        raise ValueError(f"no rows for splits={splits!r}")
    case_ids = [row["case_id"] for row in rows]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("duplicate case_id in selected evaluation rows")
    rows.sort(key=lambda row: row["case_id"])
    if limit > 0:
        random.Random(seed).shuffle(rows)
        rows = rows[:limit]
        rows.sort(key=lambda row: row["case_id"])
    selected_total = len(rows)
    return rows[shard_index::num_shards], selected_total


def _atomic_nifti(
    path: Path,
    values: np.ndarray,
    reference: nib.spatialimages.SpatialImage,
    *,
    attempts: int = 5,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = reference.header.copy()
    header.set_data_dtype(np.float32)
    image = nib.Nifti1Image(
        np.asarray(values, dtype=np.float32),
        reference.affine,
        header=header,
    )
    last_error: OSError | None = None
    for attempt in range(1, attempts + 1):
        temporary = path.with_name(
            path.name.removesuffix(".nii.gz")
            + f".{os.getpid()}.{attempt}.tmp.nii.gz"
        )
        try:
            nib.save(image, temporary)
            os.replace(temporary, path)
            return
        except OSError as error:
            last_error = error
            temporary.unlink(missing_ok=True)
            if error.errno not in {
                errno.EIO,
                errno.ESTALE,
                errno.ETIMEDOUT,
                errno.EAGAIN,
            } or attempt == attempts:
                raise
            time.sleep(min(2**attempt, 16))
    raise RuntimeError(f"failed to write NIfTI after {attempts} attempts") from last_error


def _load_ct(row: dict[str, str]) -> tuple[nib.spatialimages.SpatialImage, np.ndarray]:
    image = nib.load(row["ct_path"])
    if len(image.shape) != 3:
        raise ValueError(f"non-3D CT for {row['case_id']}: {image.shape}")
    ct = np.asanyarray(image.dataobj, dtype=np.float32)
    if not np.isfinite(ct).all():
        raise ValueError(f"non-finite CT for {row['case_id']}")
    return image, ct


def _load_anatomy(
    row: dict[str, str],
    reference: nib.spatialimages.SpatialImage,
    anatomy_label_map: dict[int, int],
) -> np.ndarray:
    path_value = row.get("fused_anatomy_path", "") or row.get(
        "anatomy_path", ""
    )
    if not path_value:
        raise ValueError(f"missing anatomy path for {row['case_id']}")
    image = nib.load(path_value)
    if image.shape != reference.shape or not np.allclose(
        image.affine, reference.affine, atol=1e-3, rtol=0
    ):
        raise ValueError(
            f"anatomy/CT geometry mismatch for {row['case_id']}: "
            f"{image.shape} != {reference.shape}"
        )
    anatomy = np.asanyarray(image.dataobj)
    if anatomy.ndim != 3 or not np.isfinite(anatomy).all():
        raise ValueError(f"invalid anatomy for {row['case_id']}")
    rounded = np.rint(anatomy)
    if not np.allclose(anatomy, rounded, atol=1e-4, rtol=0):
        raise ValueError(f"non-integer anatomy labels for {row['case_id']}")
    if rounded.min() < 0:
        raise ValueError(f"negative anatomy labels for {row['case_id']}")
    observed = {int(value) for value in np.unique(rounded)}
    unknown = observed.difference({0}, anatomy_label_map)
    if unknown:
        raise ValueError(
            f"unknown anatomy labels for {row['case_id']}: {sorted(unknown)}"
        )
    maximum_source = max(anatomy_label_map, default=0)
    lookup = np.zeros(maximum_source + 1, dtype=np.int32)
    for source_id, model_id in anatomy_label_map.items():
        lookup[source_id] = model_id
    return lookup[rounded.astype(np.int64)].astype(np.float32)


def _parse_spacing(value: str) -> tuple[float, float, float]:
    parts = tuple(float(part) for part in value.replace("x", ",").split(","))
    if len(parts) != 3 or any(part <= 0 for part in parts):
        raise ValueError(f"expected positive spacing triplet, got {value!r}")
    return parts


def anatomy_target_source_ids(
    ontology_rows: list[dict[str, str]], target_host: str
) -> frozenset[int]:
    return frozenset(
        int(row["label_id"])
        for row in ontology_rows
        if target_host in (
            row.get("target_host", "").strip().lower(),
            row.get("label_name", "").strip().lower(),
        )
    )


def _load_distribution_condition(
    root: Path,
    case_id: str,
    *,
    num_organ_ids: int = 223,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    path = root / f"{case_id}.npz"
    if not path.is_file():
        raise FileNotFoundError(
            f"CT-only distribution condition missing for {case_id}: {path}"
        )
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "case_id",
            "model_label_ids",
            "predicted_pet_probability",
            "checkpoint_sha256",
        }
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(
                f"condition {path} missing arrays {sorted(missing)}"
            )
        stored_case_id = str(archive["case_id"].item())
        organ_ids = np.asarray(
            archive["model_label_ids"],
            dtype=np.int64,
        )
        probability = np.asarray(
            archive["predicted_pet_probability"],
            dtype=np.float32,
        )
        condition_checkpoint_sha256 = str(
            archive["checkpoint_sha256"].item()
        )
    if stored_case_id != case_id:
        raise ValueError(f"condition identity mismatch: {path}")
    if (
        organ_ids.ndim != 1
        or probability.shape != (len(organ_ids), 128)
        or np.any(organ_ids <= 0)
        or np.any(organ_ids >= num_organ_ids)
        or len(set(organ_ids.tolist())) != len(organ_ids)
        or not np.isfinite(probability).all()
        or probability.min(initial=0.0) < 0.0
        or probability.max(initial=0.0) > 1.0 + 1e-5
        or not np.allclose(
            probability.sum(axis=1),
            np.ones(len(organ_ids), dtype=np.float32),
            atol=1e-4,
            rtol=1e-4,
        )
        or len(condition_checkpoint_sha256) != 64
    ):
        raise ValueError(f"invalid CT-only distribution condition: {path}")
    dense = np.zeros((num_organ_ids, 128), dtype=np.float32)
    mask = np.zeros(num_organ_ids, dtype=bool)
    dense[organ_ids] = probability
    mask[organ_ids] = True
    return (
        torch.from_numpy(dense),
        torch.from_numpy(mask),
        condition_checkpoint_sha256,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--base-checkpoint",
        type=Path,
        default=None,
        help=(
            "Frozen first-stage AVAE checkpoint. When provided, --checkpoint "
            "is interpreted as a residual AVAE+ControlNet checkpoint and the "
            "normalized predictions are added before SUV clipping."
        ),
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--paired-manifest", type=Path, required=True)
    parser.add_argument(
        "--distribution-condition-root",
        type=Path,
        default=None,
        help=(
            "Directory containing <case_id>.npz native CT-only WORM PET "
            "probabilities. Required by AVAP_CT_DISTRIBUTION_GATED_DECODER."
        ),
    )
    parser.add_argument(
        "--distribution-condition-mode",
        choices=("normal", "zero", "shuffled"),
        default="normal",
        help=(
            "Normal patient-matched condition, no-information zero ablation, "
            "or deterministic whole-patient shuffled condition."
        ),
    )
    parser.add_argument(
        "--allow-alternative-distribution-checkpoint-sha256",
        default="",
        help=(
            "Explicit ablation-only override for a frozen alternative "
            "CT-to-PET distribution checkpoint. The supplied SHA256 must "
            "match both the cache summary and every per-case artifact."
        ),
    )
    parser.add_argument(
        "--distribution-condition-case-id-prefix",
        default="",
        help=(
            "Explicit namespace prepended to paired-manifest case IDs when "
            "resolving WORM condition archives (for example autopet::)."
        ),
    )
    parser.add_argument(
        "--anatomy-label-map-csv",
        type=Path,
        default=None,
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--prediction-root",
        type=Path,
        default=None,
        help=(
            "Optional flat prediction directory. When set, generated PET is "
            "written as <prediction-root>/<case_id>.nii.gz while per-case "
            "records and shard summaries remain under --out-dir."
        ),
    )
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--splits",
        default="",
        help="Comma-separated split names; overrides --split when provided.",
    )
    parser.add_argument("--limit-cases", type=int, default=1000)
    parser.add_argument("--expected-total-cases", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--roi-size", default="128,128,128")
    parser.add_argument("--overlap", type=float, default=0.50)
    parser.add_argument("--sw-batch-size", type=int, default=1)
    parser.add_argument("--expected-spacing", default="")
    parser.add_argument("--expected-grid-contract", default="")
    parser.add_argument(
        "--crop-target-host",
        default="",
        help=(
            "Optional canonical anatomy target_host to crop before sliding-"
            "window inference, for example brain. The crop is derived only "
            "from fused anatomy and is written back into source geometry."
        ),
    )
    parser.add_argument(
        "--crop-margin-voxels",
        type=int,
        default=16,
        help="Symmetric context margin around --crop-target-host.",
    )
    parser.add_argument(
        "--mask-output-to-crop-target-host",
        action="store_true",
        help=(
            "Set output outside the selected target-host anatomy union to "
            "zero. Requires --crop-target-host."
        ),
    )
    parser.add_argument(
        "--skip-missing-crop-target-host",
        action="store_true",
        help=(
            "Record a non-evaluable skip instead of failing when the selected "
            "target host is absent from the fused-anatomy field of view."
        ),
    )
    parser.add_argument(
        "--disable-save-mem-cache-flush",
        action="store_true",
        help=(
            "Disable MAISI block-level torch.cuda.empty_cache() calls during "
            "inference. This does not change tensor operations or weights and "
            "is intended for high-memory GPUs where repeated allocator cache "
            "flushes are unnecessary."
        ),
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if not 0 <= args.overlap < 1:
        raise ValueError("--overlap must be in [0,1)")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard contract")
    if args.limit_cases < 0:
        raise ValueError("--limit-cases must be non-negative")
    if args.sw_batch_size < 1:
        raise ValueError("--sw-batch-size must be positive")
    crop_target_host = str(args.crop_target_host).strip().lower()
    if args.crop_margin_voxels < 0:
        raise ValueError("--crop-margin-voxels must be non-negative")
    if args.mask_output_to_crop_target_host and not crop_target_host:
        raise ValueError(
            "--mask-output-to-crop-target-host requires --crop-target-host"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.prediction_root is not None:
        args.prediction_root.mkdir(parents=True, exist_ok=True)
    splits = tuple(
        part.strip()
        for part in (args.splits or args.split).split(",")
        if part.strip()
    )
    if not splits:
        raise ValueError("at least one split is required")
    rows, selected_total = load_rows(
        args.paired_manifest,
        splits,
        args.seed,
        args.limit_cases,
        args.num_shards,
        args.shard_index,
    )
    if selected_total != args.expected_total_cases:
        raise ValueError(
            f"selected_total={selected_total}, expected={args.expected_total_cases}"
        )

    with args.config.open(encoding="utf-8") as handle:
        config = json.load(handle)
    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    checkpoint_args = checkpoint.get("args", {})
    model_family = checkpoint.get("model_family")
    legacy_avap_controlnet = model_family == "AVAP_CONTROLNET"
    joint_log_avap = model_family == "AVAP_JOINT_LOG1P_CONTROLNET"
    distribution_gated_avap = (
        model_family == "AVAP_CT_DISTRIBUTION_GATED_DECODER"
    )
    fullres_dilated_fcn = (
        model_family == "AVAP_FULLRES_DILATED_CONDITIONAL_FCN"
    )
    distribution_conditioned_model = (
        distribution_gated_avap or fullres_dilated_fcn
    )
    log_domain_avap = (
        legacy_avap_controlnet
        or joint_log_avap
        or distribution_gated_avap
        or fullres_dilated_fcn
    )
    residual_cascade = args.base_checkpoint is not None and not log_domain_avap
    if (
        joint_log_avap or distribution_conditioned_model
    ) and args.base_checkpoint is not None:
        raise ValueError(
            "joint log-domain AVAP inference must not add a base model"
        )
    if not (residual_cascade or log_domain_avap) and not bool(
        checkpoint_args.get("variational", False)
    ):
        raise ValueError("checkpoint is not a variational CT-to-PET model")
    anatomy_controlnet = log_domain_avap or residual_cascade or bool(
        checkpoint_args.get("anatomy_controlnet", False)
    )
    if distribution_conditioned_model:
        if args.distribution_condition_root is None:
            raise ValueError(
                "--distribution-condition-root is required for the "
                "distribution-gated AVAP"
            )
        condition_summary = (
            args.distribution_condition_root.parent / "summary.json"
        )
        if not condition_summary.is_file():
            raise FileNotFoundError(
                f"condition-cache summary missing: {condition_summary}"
            )
        condition_contract = json.loads(
            condition_summary.read_text(encoding="utf-8")
        )
        expected_distribution_sha256 = str(
            checkpoint_args.get("distribution_checkpoint_sha256", "")
        ).lower()
        alternative_distribution_sha256 = str(
            args.allow_alternative_distribution_checkpoint_sha256
        ).strip().lower()
        accepted_distribution_sha256 = (
            alternative_distribution_sha256
            or expected_distribution_sha256
        )
        if (
            condition_contract.get("status")
            != "WORM_SHARED_ENCODER_PET_PROBABILITY_CACHE_COMPLETE"
            or condition_contract.get("pet_input", "invalid") is not None
            or not expected_distribution_sha256
            or len(accepted_distribution_sha256) != 64
            or condition_contract.get("distribution_checkpoint_sha256")
            != accepted_distribution_sha256
        ):
            raise ValueError("invalid CT-only WORM condition-cache contract")
    elif args.distribution_condition_root is not None:
        raise ValueError(
            "--distribution-condition-root is only valid for the "
            "distribution-gated AVAP"
        )
    if fullres_dilated_fcn:
        model = FullResolutionDilatedConditionalFCN(
            channels_last_3d=bool(
                checkpoint_args.get("channels_last_3d", False)
            )
        )
        if bool(checkpoint_args.get("channels_last_3d", False)):
            model.to(memory_format=torch.channels_last_3d)
    else:
        codec = build_autoencoder(config["autoencoder_def"], num_splits=1)
    if fullres_dilated_fcn:
        pass
    elif distribution_gated_avap:
        model = DistributionGatedAnatomyControlledTranslator(codec)
    elif anatomy_controlnet:
        model = AnatomyControlledVariationalMaisiTranslator(codec)
    else:
        model = VariationalMaisiTranslator(codec)
    model.load_state_dict(checkpoint["model"], strict=True)
    base_model = None
    base_checkpoint = None
    base_checkpoint_sha256 = None
    if residual_cascade or legacy_avap_controlnet:
        if residual_cascade and (
            checkpoint.get("loss_profile") != "residual_anatomy_quantitative"
        ):
            raise ValueError("residual checkpoint loss-profile mismatch")
        if args.base_checkpoint is None:
            raise ValueError(
                "--base-checkpoint is required for AVAP-ControlNet inference"
            )
        base_checkpoint = torch.load(
            args.base_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        if legacy_avap_controlnet:
            expected_sha256 = str(
                checkpoint_args.get("base_checkpoint_sha256", "")
            ).lower()
            expected_epoch = checkpoint_args.get("expected_base_epoch")
            expected_step = checkpoint_args.get("expected_base_global_step")
            observed_sha256 = file_sha256(args.base_checkpoint)
            if not expected_sha256 or observed_sha256 != expected_sha256:
                raise ValueError("AVAP foundation checkpoint SHA256 mismatch")
            if expected_epoch is not None and int(
                base_checkpoint.get("epoch", -1)
            ) != int(expected_epoch):
                raise ValueError("AVAP foundation checkpoint epoch mismatch")
            if expected_step is not None and int(
                base_checkpoint.get("global_step", -1)
            ) != int(expected_step):
                raise ValueError(
                    "AVAP foundation checkpoint global-step mismatch"
                )
            if float(checkpoint_args.get("suv_clip_max", 20.0)) != 20.0:
                raise ValueError("AVAP SUV clip contract must be exactly 20")
        base_codec = build_autoencoder(
            config["autoencoder_def"], num_splits=1
        )
        base_model = VariationalMaisiTranslator(base_codec)
        base_model.load_state_dict(base_checkpoint["model"], strict=True)
        base_checkpoint_sha256 = file_sha256(args.base_checkpoint)
    disabled_save_mem_modules = 0
    if args.disable_save_mem_cache_flush:
        for loaded_model in (model, base_model):
            if loaded_model is None:
                continue
            for module in loaded_model.modules():
                if hasattr(module, "save_mem"):
                    module.save_mem = False
                    disabled_save_mem_modules += 1
        if disabled_save_mem_modules == 0:
            raise RuntimeError(
                "requested save_mem cache-flush disable but no modules exposed "
                "the save_mem flag"
            )
    anatomy_label_map: dict[int, int] = {}
    crop_model_ids: frozenset[int] = frozenset()
    if anatomy_controlnet:
        if args.anatomy_label_map_csv is None:
            raise ValueError(
                "--anatomy-label-map-csv is required for anatomy ControlNet"
            )
        with args.anatomy_label_map_csv.open(
            newline="", encoding="utf-8"
        ) as handle:
            ontology_rows = list(csv.DictReader(handle))
        source_ids = [int(row["label_id"]) for row in ontology_rows]
        if not source_ids or len(source_ids) != len(set(source_ids)):
            raise ValueError("invalid anatomy label map")
        anatomy_label_map = {
            source_id: model_id
            for model_id, source_id in enumerate(source_ids, start=1)
        }
        if crop_target_host:
            crop_source_ids = anatomy_target_source_ids(ontology_rows, crop_target_host)
            if not crop_source_ids:
                raise ValueError(
                    "no anatomy labels resolve to crop target host "
                    f"{crop_target_host!r}"
                )
            crop_model_ids = frozenset(
                anatomy_label_map[source_id]
                for source_id in crop_source_ids
            )
    elif crop_target_host:
        raise ValueError(
            "--crop-target-host requires an anatomy-controlled model"
        )
    device = torch.device("cuda")
    model.to(device).eval()
    if base_model is not None:
        base_model.to(device).eval()
    roi_size = parse_triplet(args.roi_size)
    expected_spacing = (
        _parse_spacing(args.expected_spacing) if args.expected_spacing else None
    )
    checkpoint_sha256 = file_sha256(args.checkpoint)
    completed_rows: list[dict[str, Any]] = []
    started = time.time()
    current_distribution_probability: torch.Tensor | None = None
    current_distribution_mask: torch.Tensor | None = None

    condition_case_ids = [
        f"{args.distribution_condition_case_id_prefix}{row['case_id']}"
        for row in rows
    ]
    shuffled_condition_case_ids: dict[str, str] = {}
    if (
        distribution_conditioned_model
        and args.distribution_condition_mode == "shuffled"
    ):
        if len(condition_case_ids) < 2:
            raise ValueError("shuffled condition requires at least two cases")
        patient_keys = [
            row.get("patient_group_hash")
            or row.get("subject_id")
            or row["case_id"]
            for row in rows
        ]
        for index, case_id in enumerate(condition_case_ids):
            donor = None
            for offset in range(1, len(condition_case_ids)):
                candidate_index = (index + offset) % len(condition_case_ids)
                if patient_keys[candidate_index] != patient_keys[index]:
                    donor = condition_case_ids[candidate_index]
                    break
            if donor is None:
                raise ValueError(
                    "shuffled condition could not find a different patient"
                )
            shuffled_condition_case_ids[case_id] = donor

    def predictor(inputs: torch.Tensor) -> torch.Tensor:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if distribution_conditioned_model:
                if (
                    current_distribution_probability is None
                    or current_distribution_mask is None
                ):
                    raise RuntimeError(
                        "distribution condition was not loaded for this case"
                    )
                condition = current_distribution_probability.expand(
                    inputs.shape[0], -1, -1
                )
                condition_mask = current_distribution_mask.expand(
                    inputs.shape[0], -1
                )
                prediction, _, _ = model(
                    inputs[:, :1],
                    inputs[:, 1:2],
                    condition,
                    condition_mask,
                    sample=False,
                )
                prediction = prediction.float().clamp(0.0, 1.0)
            elif joint_log_avap:
                prediction, _, _ = model(
                    inputs[:, :1],
                    inputs[:, 1:2],
                    sample=False,
                )
                prediction = prediction.float().clamp(0.0, 1.0)
            elif legacy_avap_controlnet:
                base_prediction, _, _ = base_model(
                    inputs[:, :1], sample=False
                )
                controlled_prediction, _, _ = model(
                    inputs[:, :1],
                    inputs[:, 1:2],
                    sample=False,
                )
                base_suv = base_prediction.float().clamp(0.0, 1.0) * 20.0
                base_log = torch.log1p(base_suv) / math.log1p(20.0)
                control_delta = (
                    controlled_prediction.float() - base_prediction.float()
                )
                prediction = (base_log + control_delta).clamp(0.0, 1.0)
            elif residual_cascade:
                base_prediction, _, _ = base_model(
                    inputs[:, :1], sample=False
                )
                residual_prediction, _, _ = model(
                    inputs[:, :1],
                    inputs[:, 1:2],
                    sample=False,
                )
                prediction = base_prediction + residual_prediction
            elif anatomy_controlnet:
                prediction, _, _ = model(
                    inputs[:, :1],
                    inputs[:, 1:2],
                    sample=False,
                )
            else:
                prediction, _, _ = model(inputs[:, :1], sample=False)
        return prediction.float()

    for index, row in enumerate(rows, 1):
        case_id = row["case_id"]
        condition_case_id = (
            f"{args.distribution_condition_case_id_prefix}{case_id}"
        )
        condition_donor_case_id = condition_case_id
        condition_checkpoint_sha256 = None
        condition_present_organ_count = None
        if distribution_conditioned_model:
            if args.distribution_condition_mode == "shuffled":
                condition_donor_case_id = shuffled_condition_case_ids[
                    condition_case_id
                ]
            (
                condition_probability_cpu,
                condition_mask_cpu,
                condition_checkpoint_sha256,
            ) = _load_distribution_condition(
                args.distribution_condition_root,
                condition_donor_case_id,
            )
            if condition_checkpoint_sha256 != accepted_distribution_sha256:
                raise ValueError(
                    "per-case WORM checkpoint SHA256 does not match the "
                    "accepted distribution-condition contract"
                )
            condition_present_organ_count = int(
                condition_mask_cpu.sum().item()
            )
            if args.distribution_condition_mode == "zero":
                condition_probability_cpu.zero_()
                condition_mask_cpu.zero_()
            current_distribution_probability = (
                condition_probability_cpu[None].to(
                    device,
                    non_blocking=True,
                )
            )
            current_distribution_mask = condition_mask_cpu[None].to(
                device,
                non_blocking=True,
            )
        prediction_path = (
            args.prediction_root / f"{case_id}.nii.gz"
            if args.prediction_root is not None
            else args.out_dir / "predictions" / f"{case_id}.nii.gz"
        )
        case_record = args.out_dir / "cases" / f"{case_id}.json"
        failure_record = args.out_dir / "failed_cases" / f"{case_id}.json"
        if args.resume and prediction_path.is_file() and case_record.is_file():
            with case_record.open(encoding="utf-8") as handle:
                cached = json.load(handle)
            if (
                cached.get("case_id") != case_id
                or cached.get("generated_pet_path") != str(prediction_path)
                or cached.get("checkpoint_sha256") != checkpoint_sha256
                or cached.get("status") != "complete"
                or cached.get("crop_target_host", "") != crop_target_host
                or cached.get("crop_margin_voxels")
                != (
                    args.crop_margin_voxels
                    if crop_target_host
                    else None
                )
                or cached.get("output_masked_to_crop_target_host")
                is not args.mask_output_to_crop_target_host
                or (
                    log_domain_avap
                    and (
                        cached.get("avap_controlnet") is not True
                        or cached.get("prediction_domain")
                        != "log1p_SUV_then_exact_inverse"
                        or cached.get("joint_log_domain_avap")
                        is not joint_log_avap
                        or cached.get("distribution_gated_avap")
                        is not distribution_gated_avap
                        or cached.get("fullres_dilated_fcn")
                        is not fullres_dilated_fcn
                        or (
                            distribution_conditioned_model
                            and (
                                cached.get("distribution_condition_mode")
                                != args.distribution_condition_mode
                                or cached.get(
                                    "distribution_condition_donor_case_id"
                                )
                                != condition_donor_case_id
                                or cached.get(
                                    "distribution_checkpoint_sha256"
                                )
                                != condition_checkpoint_sha256
                            )
                        )
                        or (
                            legacy_avap_controlnet
                            and cached.get("base_checkpoint_sha256")
                            != base_checkpoint_sha256
                        )
                    )
                )
            ):
                raise ValueError(f"invalid cached case contract: {case_record}")
            failure_record.unlink(missing_ok=True)
            completed_rows.append(cached)
            print(
                "PROGRESS "
                + json.dumps(
                    {
                        "status": "skip_complete",
                        "processed": index,
                        "shard_cases": len(rows),
                        "selected_total": selected_total,
                        "case_id": case_id,
                        "shard_index": args.shard_index,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            continue

        atomic_json(
            failure_record,
            {
                "status": "inflight_or_failed",
                "case_id": case_id,
                "shard_index": args.shard_index,
                "started_at_unix": time.time(),
            },
        )
        if args.expected_grid_contract and (
            row.get("preprocessed_grid_contract") != args.expected_grid_contract
        ):
            raise ValueError(
                "grid contract mismatch for "
                f"{case_id}: {row.get('preprocessed_grid_contract')!r}"
            )
        reference, ct = _load_ct(row)
        observed_spacing = tuple(
            float(value) for value in reference.header.get_zooms()[:3]
        )
        if expected_spacing is not None and not np.allclose(
            observed_spacing,
            expected_spacing,
            atol=1e-3,
            rtol=0,
        ):
            raise ValueError(
                f"spacing contract mismatch for {case_id}: {observed_spacing}"
            )
        ct01 = ct_normalize(
            ct, hu_min=-1000.0, hu_max=1000.0
        ).astype(np.float32)
        if anatomy_controlnet:
            anatomy = _load_anatomy(
                row, reference, anatomy_label_map
            )
            inputs_np = np.stack((ct01, anatomy), axis=0)
        else:
            anatomy = None
            inputs_np = ct01[None]
        crop_slices = tuple(slice(0, size) for size in reference.shape)
        crop_shape = tuple(int(size) for size in reference.shape)
        crop_focus_mask = None
        if crop_target_host:
            if anatomy is None:
                raise RuntimeError("crop target host requires anatomy")
            focus_mask = np.isin(anatomy, tuple(crop_model_ids))
            coordinates = np.nonzero(focus_mask)
            if not coordinates[0].size:
                if args.skip_missing_crop_target_host:
                    skipped_record = {
                        "status": "skipped_missing_crop_target_host",
                        "case_id": case_id,
                        "patient_group_hash": row.get(
                            "patient_group_hash", ""
                        ),
                        "split": row["split"],
                        "ct_path": row["ct_path"],
                        "pet_path": row.get("pet_path", ""),
                        "anatomy_path": row.get(
                            "anatomy_path",
                            row.get("fused_anatomy_path", ""),
                        ),
                        "crop_target_host": crop_target_host,
                        "crop_margin_voxels": args.crop_margin_voxels,
                        "output_masked_to_crop_target_host": (
                            args.mask_output_to_crop_target_host
                        ),
                        "reason": "target_host_absent_from_fused_anatomy_fov",
                        "checkpoint": str(args.checkpoint),
                        "checkpoint_sha256": checkpoint_sha256,
                        "checkpoint_epoch": checkpoint.get("epoch"),
                        "checkpoint_global_step": checkpoint.get(
                            "global_step"
                        ),
                    }
                    atomic_json(case_record, skipped_record)
                    failure_record.unlink(missing_ok=True)
                    completed_rows.append(skipped_record)
                    atomic_csv(
                        args.out_dir / "case_manifest.csv",
                        completed_rows,
                    )
                    elapsed = time.time() - started
                    progress = {
                        "status": "skip_missing_crop_target_host",
                        "processed": index,
                        "shard_cases": len(rows),
                        "selected_total": selected_total,
                        "case_id": case_id,
                        "shard_index": args.shard_index,
                        "elapsed_seconds": elapsed,
                    }
                    atomic_json(
                        args.out_dir / "progress.json",
                        progress,
                    )
                    print(
                        "PROGRESS "
                        + json.dumps(progress, sort_keys=True),
                        flush=True,
                    )
                    continue
                raise ValueError(
                    f"crop target host {crop_target_host!r} absent for "
                    f"{case_id}"
                )
            lower = [
                max(
                    0,
                    int(axis.min()) - args.crop_margin_voxels,
                )
                for axis in coordinates
            ]
            upper = [
                min(
                    int(reference.shape[dimension]),
                    int(axis.max()) + 1 + args.crop_margin_voxels,
                )
                for dimension, axis in enumerate(coordinates)
            ]
            crop_slices = tuple(
                slice(start, stop) for start, stop in zip(lower, upper)
            )
            inputs_np = np.ascontiguousarray(inputs_np[(slice(None), *crop_slices)])
            crop_focus_mask = np.ascontiguousarray(focus_mask[crop_slices])
            crop_shape = tuple(int(value) for value in crop_focus_mask.shape)
        inputs = torch.from_numpy(inputs_np)[None].to(device)
        torch.cuda.reset_peak_memory_stats(device)
        case_started = time.time()
        with torch.inference_mode():
            prediction01 = sliding_window_inference(
                inputs,
                roi_size=roi_size,
                sw_batch_size=args.sw_batch_size,
                predictor=predictor,
                overlap=args.overlap,
                mode="gaussian",
                progress=False,
            )
        crop_prediction_normalized = (
            prediction01[0, 0].cpu().numpy().astype(np.float32)
        )
        if log_domain_avap:
            crop_raw_prediction = np.expm1(
                crop_prediction_normalized * np.float32(math.log1p(20.0))
            )
        else:
            crop_raw_prediction = crop_prediction_normalized * 20.0
        raw_prediction = np.zeros(reference.shape, dtype=np.float32)
        raw_prediction[crop_slices] = crop_raw_prediction
        if (
            args.mask_output_to_crop_target_host
            and crop_focus_mask is not None
        ):
            masked_crop = raw_prediction[crop_slices]
            masked_crop[~crop_focus_mask] = 0.0
        prediction = np.clip(raw_prediction, 0.0, 20.0)
        peak_allocated_gib = torch.cuda.max_memory_allocated(device) / 2**30
        peak_reserved_gib = torch.cuda.max_memory_reserved(device) / 2**30
        _atomic_nifti(prediction_path, prediction, reference)
        if nib.load(prediction_path).shape != reference.shape:
            raise RuntimeError(f"written geometry mismatch for {case_id}")
        record = {
            "status": "complete",
            "case_id": case_id,
            "patient_group_hash": row.get("patient_group_hash", ""),
            "split": row["split"],
            "ct_path": row["ct_path"],
            "pet_path": row["pet_path"],
            "anatomy_path": row.get(
                "anatomy_path",
                row.get("fused_anatomy_path", ""),
            ),
            "generated_pet_path": str(prediction_path),
            "shape": "x".join(str(value) for value in reference.shape),
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_epoch": checkpoint.get("epoch"),
            "checkpoint_global_step": checkpoint.get("global_step"),
            "model_family": model_family or "",
            "avap_controlnet": log_domain_avap,
            "joint_log_domain_avap": joint_log_avap,
            "distribution_gated_avap": distribution_gated_avap,
            "fullres_dilated_fcn": fullres_dilated_fcn,
            "distribution_condition_mode": (
                args.distribution_condition_mode
                if distribution_conditioned_model
                else None
            ),
            "distribution_condition_case_id": (
                condition_case_id if distribution_conditioned_model else None
            ),
            "distribution_condition_donor_case_id": (
                condition_donor_case_id
                if distribution_conditioned_model
                else None
            ),
            "distribution_condition_present_organs": (
                condition_present_organ_count
                if distribution_conditioned_model
                else None
            ),
            "distribution_checkpoint_sha256": (
                condition_checkpoint_sha256
                if distribution_conditioned_model
                else None
            ),
            "training_distribution_checkpoint_sha256": (
                expected_distribution_sha256
                if distribution_conditioned_model
                else None
            ),
            "alternative_distribution_checkpoint_ablation": (
                bool(alternative_distribution_sha256)
                if distribution_conditioned_model
                else False
            ),
            "prediction_domain": (
                "log1p_SUV_then_exact_inverse"
                if log_domain_avap
                else "linear_SUV_over_20"
            ),
            "residual_cascade": residual_cascade,
            "base_checkpoint": (
                str(args.base_checkpoint)
                if residual_cascade or legacy_avap_controlnet
                else None
            ),
            "base_checkpoint_sha256": base_checkpoint_sha256,
            "base_checkpoint_epoch": (
                base_checkpoint.get("epoch") if base_checkpoint else None
            ),
            "deterministic_posterior_mean": True,
            "anatomy_controlnet": anatomy_controlnet,
            "suv_clip": "0,20",
            "roi_size": "x".join(str(value) for value in roi_size),
            "overlap": args.overlap,
            "sliding_window_mode": "gaussian",
            "crop_target_host": crop_target_host,
            "crop_margin_voxels": (
                args.crop_margin_voxels if crop_target_host else None
            ),
            "crop_start": [
                int(part.start) for part in crop_slices
            ],
            "crop_stop": [
                int(part.stop) for part in crop_slices
            ],
            "crop_shape": list(crop_shape),
            "output_masked_to_crop_target_host": (
                args.mask_output_to_crop_target_host
            ),
            "sw_batch_size": args.sw_batch_size,
            "save_mem_cache_flush_disabled": (
                args.disable_save_mem_cache_flush
            ),
            "save_mem_modules_disabled": disabled_save_mem_modules,
            "peak_allocated_gib": peak_allocated_gib,
            "peak_reserved_gib": peak_reserved_gib,
            "spacing_mm": list(observed_spacing),
            "grid_contract": row.get("preprocessed_grid_contract", ""),
            "raw_below_zero_fraction": float(np.mean(raw_prediction < 0)),
            "raw_above_twenty_fraction": float(np.mean(raw_prediction > 20)),
            "case_seconds": time.time() - case_started,
            "file_bytes": prediction_path.stat().st_size,
        }
        atomic_json(case_record, record)
        failure_record.unlink(missing_ok=True)
        completed_rows.append(record)
        atomic_csv(args.out_dir / "case_manifest.csv", completed_rows)
        elapsed = time.time() - started
        progress = {
            "status": "running",
            "processed": index,
            "shard_cases": len(rows),
            "selected_total": selected_total,
            "case_id": case_id,
            "shard_index": args.shard_index,
            "elapsed_seconds": elapsed,
            "cases_per_hour": index / max(elapsed, 1e-9) * 3600,
            "gpu_peak_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
        }
        atomic_json(args.out_dir / "progress.json", progress)
        print("PROGRESS " + json.dumps(progress, sort_keys=True), flush=True)

    atomic_csv(args.out_dir / "case_manifest.csv", completed_rows)
    generated_count = sum(
        row.get("status") == "complete" for row in completed_rows
    )
    skipped_missing_count = sum(
        row.get("status") == "skipped_missing_crop_target_host"
        for row in completed_rows
    )
    summary = {
        "status": "complete",
        "n_cases": len(completed_rows),
        "n_generated": generated_count,
        "n_skipped_missing_crop_target_host": skipped_missing_count,
        "selected_total": selected_total,
        "splits": list(splits),
        "seed": args.seed,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_global_step": checkpoint.get("global_step"),
        "model_family": model_family or "",
        "avap_controlnet": log_domain_avap,
        "joint_log_domain_avap": joint_log_avap,
        "distribution_gated_avap": distribution_gated_avap,
        "fullres_dilated_fcn": fullres_dilated_fcn,
        "distribution_condition_root": (
            str(args.distribution_condition_root)
            if distribution_conditioned_model
            else None
        ),
        "distribution_condition_mode": (
            args.distribution_condition_mode
            if distribution_conditioned_model
            else None
        ),
        "training_distribution_checkpoint_sha256": (
            expected_distribution_sha256
            if distribution_conditioned_model
            else None
        ),
        "accepted_distribution_checkpoint_sha256": (
            accepted_distribution_sha256
            if distribution_conditioned_model
            else None
        ),
        "alternative_distribution_checkpoint_ablation": (
            bool(alternative_distribution_sha256)
            if distribution_conditioned_model
            else False
        ),
        "distribution_condition_case_id_prefix": (
            args.distribution_condition_case_id_prefix
            if distribution_conditioned_model
            else None
        ),
        "prediction_domain": (
            "log1p_SUV_then_exact_inverse"
            if log_domain_avap
            else "linear_SUV_over_20"
        ),
        "residual_cascade": residual_cascade,
        "base_checkpoint": (
            str(args.base_checkpoint)
            if residual_cascade or legacy_avap_controlnet
            else None
        ),
        "base_checkpoint_sha256": base_checkpoint_sha256,
        "base_checkpoint_epoch": (
            base_checkpoint.get("epoch") if base_checkpoint else None
        ),
        "roi_size": roi_size,
        "overlap": args.overlap,
        "sliding_window_mode": "gaussian",
        "crop_target_host": crop_target_host,
        "crop_margin_voxels": (
            args.crop_margin_voxels if crop_target_host else None
        ),
        "output_masked_to_crop_target_host": (
            args.mask_output_to_crop_target_host
        ),
        "sw_batch_size": args.sw_batch_size,
        "save_mem_cache_flush_disabled": args.disable_save_mem_cache_flush,
        "save_mem_modules_disabled": disabled_save_mem_modules,
        "expected_spacing_mm": (
            list(expected_spacing) if expected_spacing is not None else None
        ),
        "expected_grid_contract": args.expected_grid_contract,
        "prediction_root": (
            str(args.prediction_root)
            if args.prediction_root is not None
            else str(args.out_dir / "predictions")
        ),
        "deterministic_posterior_mean": True,
        "anatomy_controlnet": anatomy_controlnet,
        "suv_clip": [0.0, 20.0],
        "elapsed_seconds": time.time() - started,
    }
    atomic_json(args.out_dir / "summary.json", summary)
    atomic_json(args.out_dir / "progress.json", summary)
    print("SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
