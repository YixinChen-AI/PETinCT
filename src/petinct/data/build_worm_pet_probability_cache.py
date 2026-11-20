"""Build CT-only WORM PET-probability conditions for AVAP.

The builder reads only the CT organ-distribution library and the frozen shared
AVAP-encoder ROI feature library.  Real PET is neither loaded nor accepted as
an argument.  Each output contains WORM's native 128-bin probability for every
present organ and is therefore directly consumable by the gated AVAP decoder.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from petinct.data.organ_relation_dataset import cdf_to_probability
from petinct.models.ct_to_pet_distribution import (
    CtToPetOrganDistributionNet,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_manifest(
    root: Path,
    *,
    path_column: str,
    required_extra: set[str],
) -> dict[str, dict[str, str]]:
    path = root / "case_manifest.csv"
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "case_id",
            "patient_group_hash",
            "split",
            "n_present_organs",
            path_column,
            *required_extra,
        }
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} missing columns {sorted(missing)}")
        rows = list(reader)
    case_ids = [row["case_id"] for row in rows]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError(f"duplicate case IDs in {path}")
    return {row["case_id"]: row for row in rows}


def _load_case(
    ct_root: Path,
    ct_row: dict[str, str],
    roi_root: Path | None = None,
    roi_row: dict[str, str] | None = None,
) -> dict[str, Any]:
    ct_path = ct_root / ct_row["distribution_path"]
    with np.load(ct_path, allow_pickle=False) as archive:
        required = {
            "case_id",
            "patient_group_hash",
            "split",
            "model_label_ids",
            "cdf_128",
        }
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"{ct_path} missing {sorted(missing)}")
        ct_identity = {
            key: str(archive[key].item())
            for key in ("case_id", "patient_group_hash", "split")
        }
        organ_ids = np.asarray(archive["model_label_ids"], dtype=np.int64)
        ct_probability = cdf_to_probability(archive["cdf_128"])
    expected_identity = {
        key: ct_row[key]
        for key in ("case_id", "patient_group_hash", "split")
    }
    if ct_identity != expected_identity:
        raise ValueError(f"archive identity mismatch for {ct_row['case_id']}")
    if (
        organ_ids.ndim != 1
        or np.any(organ_ids <= 0)
        or len(set(organ_ids.tolist())) != len(organ_ids)
        or ct_probability.shape != (len(organ_ids), 128)
        or not np.isfinite(ct_probability).all()
    ):
        raise ValueError(f"invalid CT case {ct_row['case_id']}")
    result = {
        "case_id": ct_row["case_id"],
        "patient_group_hash": ct_row["patient_group_hash"],
        "split": ct_row["split"],
        "organ_ids": torch.from_numpy(organ_ids),
        "ct_probability": torch.from_numpy(ct_probability),
    }
    if (roi_root is None) != (roi_row is None):
        raise ValueError("ROI root and row must be provided together")
    if roi_root is None or roi_row is None:
        result["ct_roi_features"] = None
        return result

    locked = ("case_id", "patient_group_hash", "split", "n_present_organs")
    for key in locked:
        if ct_row[key] != roi_row[key]:
            raise ValueError(
                f"CT/ROI manifest mismatch {ct_row['case_id']} {key}"
            )
    roi_path = roi_root / roi_row["feature_path"]
    with np.load(roi_path, allow_pickle=False) as archive:
        required = {
            "case_id",
            "patient_group_hash",
            "split",
            "model_label_ids",
            "roi_features",
            "organ_feature_counts",
        }
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"{roi_path} missing {sorted(missing)}")
        roi_identity = {
            key: str(archive[key].item())
            for key in ("case_id", "patient_group_hash", "split")
        }
        roi_ids = np.asarray(archive["model_label_ids"], dtype=np.int64)
        roi_features = np.asarray(archive["roi_features"], dtype=np.float32)
        roi_counts = np.asarray(
            archive["organ_feature_counts"],
            dtype=np.float32,
        )
    if roi_identity != expected_identity:
        raise ValueError(f"ROI identity mismatch for {ct_row['case_id']}")
    if (
        not np.array_equal(organ_ids, roi_ids)
        or roi_features.ndim != 2
        or roi_features.shape[0] != len(organ_ids)
        or roi_counts.shape != (len(organ_ids),)
        or not np.isfinite(roi_features).all()
        or not np.isfinite(roi_counts).all()
        or np.any(roi_counts < 0)
        or np.any(
            (roi_counts == 0)
            & np.any(np.abs(roi_features) > 1e-6, axis=1)
        )
    ):
        raise ValueError(f"invalid CT ROI case {ct_row['case_id']}")
    result["ct_roi_features"] = torch.from_numpy(roi_features)
    return result


def _collate(examples: list[dict[str, Any]]) -> dict[str, Any]:
    maximum = max(len(item["organ_ids"]) for item in examples)
    has_roi = examples[0]["ct_roi_features"] is not None
    if any((item["ct_roi_features"] is not None) != has_roi for item in examples):
        raise ValueError("ROI feature availability changes within batch")
    feature_dim = (
        int(examples[0]["ct_roi_features"].shape[1]) if has_roi else None
    )
    batch = len(examples)
    organ_ids = torch.zeros((batch, maximum), dtype=torch.long)
    organ_mask = torch.zeros((batch, maximum), dtype=torch.bool)
    probability = torch.zeros((batch, maximum, 128), dtype=torch.float32)
    roi = (
        torch.zeros(
            (batch, maximum, feature_dim),
            dtype=torch.float32,
        )
        if feature_dim is not None
        else None
    )
    for index, item in enumerate(examples):
        count = len(item["organ_ids"])
        if (
            feature_dim is not None
            and item["ct_roi_features"].shape != (count, feature_dim)
        ):
            raise ValueError("ROI feature dimension changes within batch")
        organ_ids[index, :count] = item["organ_ids"]
        organ_mask[index, :count] = True
        probability[index, :count] = item["ct_probability"]
        if roi is not None:
            roi[index, :count] = item["ct_roi_features"]
    return {
        "organ_ids": organ_ids,
        "organ_mask": organ_mask,
        "ct_probability": probability,
        "ct_roi_features": roi,
    }


def _model_arguments(checkpoint: dict[str, Any]) -> dict[str, Any]:
    arguments = dict(
        checkpoint.get("run_contract", {}).get("arguments", {})
    )
    required = {
        "num_organ_ids",
        "attention_layers",
        "attention_heads",
        "feedforward_dim",
        "context_mode",
    }
    missing = required.difference(arguments)
    if missing:
        raise ValueError(
            f"WORM checkpoint arguments missing {sorted(missing)}"
        )
    state = checkpoint.get("model", {})
    if "organ_id_dim" not in arguments:
        weight = state.get("organ_id_embedding.weight")
        if weight is None or weight.ndim != 2:
            raise ValueError("cannot infer historical organ_id_dim")
        arguments["organ_id_dim"] = int(weight.shape[1])
    if "token_dim" not in arguments:
        weight = state.get("token_projection.0.weight")
        if weight is None or weight.ndim != 2:
            raise ValueError("cannot infer historical token_dim")
        arguments["token_dim"] = int(weight.shape[0])
    return arguments


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ct-library-root", type=Path, required=True)
    parser.add_argument("--ct-roi-library-root", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--splits", default="train,val")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()
    if (
        args.batch_size < 1
        or args.progress_every < 1
        or args.max_cases < 0
    ):
        raise ValueError("invalid cache-builder arguments")
    if not torch.cuda.is_available():
        raise RuntimeError("scheduled CUDA node required")
    observed_sha = _sha256(args.checkpoint)
    if observed_sha != args.checkpoint_sha256.lower():
        raise RuntimeError("WORM checkpoint SHA256 mismatch")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    splits = {value.strip() for value in args.splits.split(",") if value.strip()}
    if not splits:
        raise ValueError("at least one split is required")
    ct_rows = _read_manifest(
        args.ct_library_root,
        path_column="distribution_path",
        required_extra=set(),
    )
    selected_case_ids = {
        case_id
        for case_id, row in ct_rows.items()
        if row["split"] in splits
    }
    roi_rows: dict[str, dict[str, str]] | None = None
    if args.ct_roi_library_root is not None:
        roi_rows = _read_manifest(
            args.ct_roi_library_root,
            path_column="feature_path",
            required_extra={"feature_dim"},
        )
        roi_selected_case_ids = {
            case_id
            for case_id, row in roi_rows.items()
            if row["split"] in splits
        }
        if selected_case_ids != roi_selected_case_ids:
            raise ValueError(
                "CT/ROI selected-split case sets differ: "
                f"ct_only={len(selected_case_ids.difference(roi_selected_case_ids))} "
                f"roi_only={len(roi_selected_case_ids.difference(selected_case_ids))}"
            )
    selected = [
        case_id
        for case_id in sorted(selected_case_ids)
    ]
    if args.max_cases:
        selected = selected[: args.max_cases]
    if not selected:
        raise ValueError("no cases selected")
    roi_feature_dim: int | None = None
    if roi_rows is not None:
        feature_dims = {
            int(roi_rows[case_id]["feature_dim"]) for case_id in selected
        }
        if len(feature_dims) != 1:
            raise ValueError(f"ROI feature dimensions differ: {feature_dims}")
        roi_feature_dim = feature_dims.pop()

    device = torch.device("cuda", 0)
    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    # Historical CT-to-PET distribution checkpoints predate the canonical
    # WORM-Net display name.  Scientific identity is therefore established by
    # the frozen run-contract arguments plus strict state-dict loading below,
    # rather than by a presentation-only model_name string.
    checkpoint_has_roi = any(
        key == "roi_gate_logit" or key.startswith("roi_projection.")
        for key in checkpoint["model"]
    )
    if checkpoint_has_roi != (roi_feature_dim is not None):
        raise RuntimeError(
            "checkpoint/cache ROI contract mismatch: "
            f"checkpoint_has_roi={checkpoint_has_roi} "
            f"roi_library_provided={roi_feature_dim is not None}"
        )
    model_args = _model_arguments(checkpoint)
    if str(model_args["context_mode"]) not in {
        "full",
        "local",
        "organ_id_only",
    }:
        raise RuntimeError("unsupported CT-to-PET context mode")
    decoder_weight = checkpoint["model"].get(
        "pet_distribution_decoder.4.weight"
    )
    if decoder_weight is None or decoder_weight.ndim != 2:
        raise RuntimeError("checkpoint lacks the terminal PET decoder weight")
    output_distribution_dim = int(decoder_weight.shape[0])
    model = CtToPetOrganDistributionNet(
        num_organ_ids=int(model_args["num_organ_ids"]),
        organ_id_dim=int(model_args["organ_id_dim"]),
        token_dim=int(model_args["token_dim"]),
        attention_layers=int(model_args["attention_layers"]),
        attention_heads=int(model_args["attention_heads"]),
        feedforward_dim=int(model_args["feedforward_dim"]),
        context_mode=str(model_args["context_mode"]),
        roi_feature_dim=roi_feature_dim,
        output_distribution_dim=output_distribution_dim,
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    args.output_dir.mkdir(parents=True, exist_ok=False)
    probability_root = args.output_dir / "predicted_probability"
    probability_root.mkdir()
    manifest_rows: list[dict[str, Any]] = []
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for start in range(0, len(selected), args.batch_size):
            case_ids = selected[start : start + args.batch_size]
            examples = [
                _load_case(
                    args.ct_library_root,
                    ct_rows[case_id],
                    args.ct_roi_library_root,
                    None if roi_rows is None else roi_rows[case_id],
                )
                for case_id in case_ids
            ]
            batch = _collate(examples)
            with torch.autocast(
                "cuda",
                dtype=torch.bfloat16,
                enabled=args.amp,
            ):
                model_kwargs: dict[str, torch.Tensor] = {}
                if batch["ct_roi_features"] is not None:
                    model_kwargs["ct_roi_features"] = batch[
                        "ct_roi_features"
                    ].to(device, non_blocking=True)
                prediction = model(
                    batch["ct_probability"].to(
                        device,
                        non_blocking=True,
                    ),
                    batch["organ_ids"].to(device, non_blocking=True),
                    batch["organ_mask"].to(device, non_blocking=True),
                    **model_kwargs,
                    output_mode="probability",
                )
            prediction = prediction.float().cpu()
            for index, example in enumerate(examples):
                count = len(example["organ_ids"])
                probability = prediction[index, :count].numpy()
                if (
                    probability.shape != (count, output_distribution_dim)
                    or not np.isfinite(probability).all()
                    or np.any(probability < 0.0)
                    or np.any(probability > 1.0 + 1e-5)
                    or not np.allclose(
                        probability.sum(axis=1),
                        np.ones(count, dtype=np.float32),
                        atol=1e-4,
                        rtol=1e-4,
                    )
                ):
                    raise RuntimeError(
                        f"invalid WORM probability for {example['case_id']}"
                    )
                relative = Path("predicted_probability") / (
                    f"{example['case_id']}.npz"
                )
                output = args.output_dir / relative
                temporary = output.with_name(
                    f".{output.name}.{os.getpid()}.tmp.npz"
                )
                np.savez_compressed(
                    temporary,
                    case_id=np.asarray(example["case_id"]),
                    patient_group_hash=np.asarray(
                        example["patient_group_hash"]
                    ),
                    split=np.asarray(example["split"]),
                    model_label_ids=example["organ_ids"].numpy().astype(
                        np.int32
                    ),
                    predicted_pet_probability=probability.astype(
                        np.float32
                    ),
                    checkpoint_sha256=np.asarray(observed_sha),
                )
                os.replace(temporary, output)
                manifest_rows.append(
                    {
                        "case_id": example["case_id"],
                        "patient_group_hash": example[
                            "patient_group_hash"
                        ],
                        "split": example["split"],
                        "n_present_organs": count,
                        "probability_path": str(relative),
                    }
                )
            completed = start + len(examples)
            batch_index = start // args.batch_size + 1
            if (
                batch_index == 1
                or batch_index % args.progress_every == 0
                or completed == len(selected)
            ):
                elapsed = max(time.monotonic() - started, 1e-9)
                print(
                    "WORM_PROBABILITY_CACHE_PROGRESS "
                    f"cases={completed}/{len(selected)} "
                    f"cases_per_s={completed / elapsed:.4f} "
                    f"gpu_peak_gib="
                    f"{torch.cuda.max_memory_allocated(device) / 2**30:.4f}",
                    flush=True,
                )
    if len(manifest_rows) != len(selected):
        raise RuntimeError("probability-cache output count mismatch")
    manifest_path = args.output_dir / "case_manifest.csv"
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "case_id",
                "patient_group_hash",
                "split",
                "n_present_organs",
                "probability_path",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)
    split_counts = {
        split: sum(row["split"] == split for row in manifest_rows)
        for split in sorted(splits)
    }
    summary = {
        "status": "WORM_SHARED_ENCODER_PET_PROBABILITY_CACHE_COMPLETE",
        "ct_library_root": str(args.ct_library_root.resolve()),
        "ct_library_manifest_sha256": _sha256(
            args.ct_library_root / "case_manifest.csv"
        ),
        "ct_roi_library_root": (
            None
            if args.ct_roi_library_root is None
            else str(args.ct_roi_library_root.resolve())
        ),
        "ct_roi_library_manifest_sha256": (
            None
            if args.ct_roi_library_root is None
            else _sha256(
                args.ct_roi_library_root / "case_manifest.csv"
            )
        ),
        "distribution_checkpoint": str(args.checkpoint.resolve()),
        "distribution_checkpoint_sha256": observed_sha,
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_global_step": int(checkpoint["global_step"]),
        "context_mode": str(model_args["context_mode"]),
        "pet_input": None,
        "condition_contract": (
            f"native {output_distribution_dim}-bin PET probability from CT organ distributions "
            "and organ IDs"
            if roi_feature_dim is None
            else f"native {output_distribution_dim}-bin PET probability with CT ROI features"
        ),
        "n_cases": len(manifest_rows),
        "splits": split_counts,
        "roi_feature_dim": roi_feature_dim,
        "output_distribution_dim": output_distribution_dim,
        "case_manifest_sha256": _sha256(manifest_path),
        "elapsed_seconds": time.monotonic() - started,
        "gpu_peak_gib": torch.cuda.max_memory_allocated(device) / 2**30,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
