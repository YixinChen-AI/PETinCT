"""Build a full-volume patient-by-organ SUV CDF library.

Each case is processed on its complete PET/anatomy grid. Background label zero
is excluded, while the set of foreground labels is discovered independently
for every case. The canonical stored representation is the unsmoothed
empirical CDF on a shared physical SUVbw grid.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


REQUIRED_MANIFEST_COLUMNS = {
    "case_id",
    "patient_group_hash",
    "pet_path",
    "anatomy_path",
    "suvbw_factor",
    "split",
}
NAME_COLUMNS = (
    "label_name",
    "anatomy_name",
    "structure_name",
    "class_name",
    "name",
)


@dataclass(frozen=True)
class DistributionContract:
    bins: int
    suv_min: float
    suv_max: float
    expected_spacing: tuple[float, float, float]
    affine_tolerance: float
    spacing_tolerance: float

    @property
    def bin_width(self) -> float:
        return (self.suv_max - self.suv_min) / self.bins

    @property
    def bin_edges(self) -> np.ndarray:
        return np.linspace(
            self.suv_min,
            self.suv_max,
            self.bins + 1,
            dtype=np.float64,
        )


def _safe_case_name(case_id: str) -> str:
    readable = re.sub(r"[^A-Za-z0-9_.-]+", "_", case_id).strip("._")
    digest = hashlib.sha256(case_id.encode("utf-8")).hexdigest()[:12]
    return f"{readable[:80] or 'case'}_{digest}.npz"


def _counts_to_cdf(counts: np.ndarray) -> np.ndarray:
    probability = counts.astype(np.float64, copy=False)
    total = float(probability.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("invalid empirical probability mass")
    probability /= total
    cdf = np.cumsum(probability, dtype=np.float64)
    cdf[-1] = 1.0
    if (
        not np.isfinite(cdf).all()
        or np.any(np.diff(cdf) < -1e-10)
        or cdf[0] < 0
        or cdf[-1] != 1.0
    ):
        raise ValueError("invalid CDF")
    return cdf.astype(np.float32)


def _load_label_map(path: str) -> tuple[dict[int, str], dict[int, int]]:
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "label_id" not in rows[0]:
        raise ValueError("label map must contain a non-empty label_id column")
    source_ids = [int(row["label_id"]) for row in rows]
    if any(label_id <= 0 for label_id in source_ids):
        raise ValueError("canonical foreground label IDs must be positive")
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("duplicate label_id in canonical label map")
    name_column = next(
        (column for column in NAME_COLUMNS if column in rows[0]),
        None,
    )
    names = {
        label_id: (
            rows[index].get(name_column, "").strip()
            if name_column is not None
            else str(label_id)
        )
        or str(label_id)
        for index, label_id in enumerate(source_ids)
    }
    model_ids = {
        source_id: model_id
        for model_id, source_id in enumerate(source_ids, start=1)
    }
    return names, model_ids


def _load_rows(path: Path, splits: set[str]) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_MANIFEST_COLUMNS.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"paired manifest missing columns {sorted(missing)}"
            )
        rows = [row for row in reader if row["split"] in splits]
    if not rows:
        raise ValueError(f"no manifest rows for splits={sorted(splits)}")
    case_ids = [row["case_id"] for row in rows]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("duplicate case_id in selected manifest rows")
    return rows


def _atomic_save_npz(path: Path, **arrays: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _read_existing(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "case_id",
            "patient_group_hash",
            "split",
            "label_ids",
            "model_label_ids",
            "label_names",
            "voxel_count",
            "volume_mm3",
            "clip_zero_fraction",
            "clip_max_fraction",
            "cdf_128",
            "spacing",
        }
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(
                f"incomplete existing case archive {path}: {sorted(missing)}"
            )
        return {key: archive[key] for key in required}


def _process_case(
    row: dict[str, str],
    output_dir: str,
    label_names: dict[int, str],
    model_label_ids: dict[int, int],
    contract: DistributionContract,
) -> dict[str, Any]:
    import nibabel as nib

    case_id = row["case_id"]
    case_path = Path(output_dir) / "cases" / _safe_case_name(case_id)
    if case_path.exists():
        archive = _read_existing(case_path)
        if str(archive["case_id"].item()) != case_id:
            raise ValueError(f"case archive ID mismatch: {case_path}")
        return {
            "status": "existing",
            "case_id": case_id,
            "case_path": str(case_path),
            "n_organs": int(archive["label_ids"].shape[0]),
        }

    pet_image = nib.load(row["pet_path"])
    anatomy_image = nib.load(row["anatomy_path"])
    if pet_image.shape != anatomy_image.shape or len(pet_image.shape) != 3:
        raise ValueError(
            f"PET/anatomy shape mismatch for {case_id}: "
            f"{pet_image.shape} vs {anatomy_image.shape}"
        )
    if not np.allclose(
        pet_image.affine,
        anatomy_image.affine,
        atol=contract.affine_tolerance,
        rtol=0.0,
    ):
        raise ValueError(f"PET/anatomy affine mismatch for {case_id}")
    spacing = tuple(float(value) for value in pet_image.header.get_zooms()[:3])
    if not np.allclose(
        spacing,
        contract.expected_spacing,
        atol=contract.spacing_tolerance,
        rtol=0.0,
    ):
        raise ValueError(
            f"spacing contract violation for {case_id}: {spacing}"
        )
    voxel_volume_mm3 = float(np.prod(spacing))

    stored_pet = np.asanyarray(pet_image.dataobj, dtype=np.float32)
    anatomy_float = np.asanyarray(anatomy_image.dataobj)
    if not np.isfinite(stored_pet).all():
        raise ValueError(f"non-finite PET for {case_id}")
    rounded = np.rint(anatomy_float)
    if not np.array_equal(anatomy_float, rounded):
        raise ValueError(f"non-integer anatomy labels for {case_id}")
    anatomy = rounded.astype(np.int64, copy=False)
    if anatomy.min(initial=0) < 0:
        raise ValueError(f"negative anatomy labels for {case_id}")

    factor = float(row["suvbw_factor"])
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError(f"invalid suvbw_factor={factor} for {case_id}")
    raw_suv = stored_pet * factor
    if not np.isfinite(raw_suv).all():
        raise ValueError(f"non-finite SUVbw for {case_id}")

    foreground = anatomy > 0
    if not foreground.any():
        raise ValueError(f"empty foreground anatomy for {case_id}")
    labels = anatomy[foreground]
    observed = np.unique(labels)
    unknown = set(int(value) for value in observed).difference(label_names)
    if unknown:
        raise ValueError(
            f"labels absent from canonical map for {case_id}: "
            f"{sorted(unknown)}"
        )
    values = raw_suv[foreground].astype(np.float64, copy=False)
    clipped = np.clip(values, contract.suv_min, contract.suv_max)
    bin_indices = np.floor(
        (clipped - contract.suv_min) / contract.bin_width
    ).astype(np.int64)
    np.clip(bin_indices, 0, contract.bins - 1, out=bin_indices)

    maximum_label = int(observed.max())
    joint = labels * contract.bins + bin_indices
    counts = np.bincount(
        joint,
        minlength=(maximum_label + 1) * contract.bins,
    ).reshape(maximum_label + 1, contract.bins)
    voxel_counts = np.bincount(labels, minlength=maximum_label + 1)
    zero_counts = np.bincount(
        labels[values <= contract.suv_min],
        minlength=maximum_label + 1,
    )
    maximum_counts = np.bincount(
        labels[values >= contract.suv_max],
        minlength=maximum_label + 1,
    )
    cdf = np.stack(
        [_counts_to_cdf(counts[int(label_id)]) for label_id in observed]
    )
    selected_counts = voxel_counts[observed].astype(np.int64)
    if np.any(selected_counts <= 0):
        raise ValueError(f"non-positive organ voxel count for {case_id}")

    label_ids = observed.astype(np.int32)
    _atomic_save_npz(
        case_path,
        case_id=np.asarray(case_id),
        patient_group_hash=np.asarray(row["patient_group_hash"]),
        split=np.asarray(row["split"]),
        label_ids=label_ids,
        model_label_ids=np.asarray(
            [model_label_ids[int(value)] for value in label_ids],
            dtype=np.int32,
        ),
        label_names=np.asarray(
            [label_names[int(value)] for value in label_ids],
            dtype=np.str_,
        ),
        voxel_count=selected_counts,
        volume_mm3=(selected_counts * voxel_volume_mm3).astype(np.float64),
        clip_zero_fraction=(
            zero_counts[observed] / selected_counts
        ).astype(np.float32),
        clip_max_fraction=(
            maximum_counts[observed] / selected_counts
        ).astype(np.float32),
        cdf_128=cdf,
        spacing=np.asarray(spacing, dtype=np.float64),
    )
    return {
        "status": "created",
        "case_id": case_id,
        "case_path": str(case_path),
        "n_organs": int(label_ids.size),
    }


def _write_manifests(
    rows: list[dict[str, str]],
    output: Path,
) -> tuple[int, int]:
    by_case = {row["case_id"]: row for row in rows}
    case_rows: list[dict[str, Any]] = []
    organ_rows: list[dict[str, Any]] = []
    for case_path in sorted((output / "cases").glob("*.npz")):
        archive = _read_existing(case_path)
        case_id = str(archive["case_id"].item())
        if case_id not in by_case:
            continue
        relative = str(case_path.relative_to(output))
        label_ids = archive["label_ids"]
        case_rows.append(
            {
                "case_id": case_id,
                "patient_group_hash": str(
                    archive["patient_group_hash"].item()
                ),
                "split": str(archive["split"].item()),
                "distribution_path": relative,
                "n_present_organs": int(label_ids.shape[0]),
            }
        )
        for index, label_id in enumerate(label_ids):
            organ_rows.append(
                {
                    "case_id": case_id,
                    "patient_group_hash": str(
                        archive["patient_group_hash"].item()
                    ),
                    "split": str(archive["split"].item()),
                    "source_label_id": int(label_id),
                    "model_label_id": int(
                        archive["model_label_ids"][index]
                    ),
                    "anatomy_name": str(archive["label_names"][index]),
                    "voxel_count": int(archive["voxel_count"][index]),
                    "volume_mm3": float(archive["volume_mm3"][index]),
                    "clip_zero_fraction": float(
                        archive["clip_zero_fraction"][index]
                    ),
                    "clip_max_fraction": float(
                        archive["clip_max_fraction"][index]
                    ),
                    "distribution_path": relative,
                    "distribution_row": index,
                }
            )
    if len(case_rows) != len(rows):
        raise RuntimeError(
            f"case archive count mismatch: {len(case_rows)} vs {len(rows)}"
        )
    for path, fieldnames, contents in (
        (
            output / "case_manifest.csv",
            [
                "case_id",
                "patient_group_hash",
                "split",
                "distribution_path",
                "n_present_organs",
            ],
            case_rows,
        ),
        (
            output / "organ_manifest.csv",
            [
                "case_id",
                "patient_group_hash",
                "split",
                "source_label_id",
                "model_label_id",
                "anatomy_name",
                "voxel_count",
                "volume_mm3",
                "clip_zero_fraction",
                "clip_max_fraction",
                "distribution_path",
                "distribution_row",
            ],
            organ_rows,
        ),
    ):
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(contents)
        os.replace(temporary, path)
    return len(case_rows), len(organ_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paired-manifest", type=Path, required=True)
    parser.add_argument("--anatomy-label-map", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--splits", default="train,val")
    parser.add_argument("--bins", type=int, default=128)
    parser.add_argument("--suv-min", type=float, default=0.0)
    parser.add_argument("--suv-max", type=float, default=20.0)
    parser.add_argument("--expected-spacing", default="1.5,1.5,3.0")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--allow-extend-completed", action="store_true")
    args = parser.parse_args()

    if args.bins != 128:
        raise ValueError("canonical organ CDF contract requires exactly 128 bins")
    if args.suv_min != 0.0 or args.suv_max != 20.0:
        raise ValueError("canonical SUVbw support must be [0,20]")
    if args.workers < 1 or args.progress_every < 1:
        raise ValueError("workers and progress-every must be positive")
    expected_spacing = tuple(
        float(value) for value in args.expected_spacing.split(",")
    )
    if len(expected_spacing) != 3:
        raise ValueError("expected-spacing must contain three values")
    contract = DistributionContract(
        bins=args.bins,
        suv_min=args.suv_min,
        suv_max=args.suv_max,
        expected_spacing=expected_spacing,
        affine_tolerance=1e-3,
        spacing_tolerance=1e-3,
    )
    splits = {value.strip() for value in args.splits.split(",") if value.strip()}
    rows = _load_rows(args.paired_manifest, splits)
    label_names, model_label_ids = _load_label_map(
        str(args.anatomy_label_map)
    )
    output = args.out_dir
    output.mkdir(parents=True, exist_ok=True)
    (output / "cases").mkdir(exist_ok=True)
    (output / "locks").mkdir(exist_ok=True)
    previous_summary = output / "summary.json"
    if previous_summary.exists():
        if not args.allow_extend_completed:
            raise FileExistsError(
                f"completed distribution library already exists: {output}"
            )
        history = output / "history"
        history.mkdir(exist_ok=True)
        archived_summary = history / (
            f"summary_before_extension_{time.strftime('%Y%m%d_%H%M%S')}.json"
        )
        os.replace(previous_summary, archived_summary)
        print(
            f"EXTEND_ARCHIVED_SUMMARY path={archived_summary}",
            flush=True,
        )

    contract_payload = {
        "status": "ORGAN_SUV_CDF128_CONTRACT",
        "paired_manifest": str(args.paired_manifest.resolve()),
        "anatomy_label_map": str(args.anatomy_label_map.resolve()),
        "splits": sorted(splits),
        "selected_cases": len(rows),
        "allow_extend_completed": bool(args.allow_extend_completed),
        "background_label": 0,
        "background_included": False,
        "label_policy": "dynamic_present_foreground_subset_of_canonical_map",
        "full_volume": True,
        "target": "physical SUVbw = stored_pet*suvbw_factor",
        "suv_clip": [contract.suv_min, contract.suv_max],
        "bins": contract.bins,
        "bin_edges": contract.bin_edges.tolist(),
        "bin_width_suv": contract.bin_width,
        "smoothing": "none",
        "training_augmentation_policy": (
            "distribution smoothing, if used, is applied only by the "
            "downstream training loader and is never persisted here"
        ),
        "stored_distribution": "cdf_128_float32",
        "histogram_stored": False,
        "probability_recovery": "first_difference_of_cdf_with_leading_zero",
        "expected_spacing_mm": list(contract.expected_spacing),
    }
    contract_path = output / "distribution_contract.json"
    contract_path.write_text(
        json.dumps(contract_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    started = time.monotonic()
    completed = 0
    created = 0
    existing = 0
    failures: list[dict[str, str]] = []
    print(
        "ORGAN_CDF_START "
        f"cases={len(rows)} workers={args.workers} bins={contract.bins} "
        "smoothing=none full_volume=1",
        flush=True,
    )
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _process_case,
                row,
                str(output),
                label_names,
                model_label_ids,
                contract,
            ): row["case_id"]
            for row in rows
        }
        for future in as_completed(futures):
            case_id = futures[future]
            completed += 1
            try:
                result = future.result()
                created += int(result["status"] == "created")
                existing += int(result["status"] == "existing")
            except Exception as error:
                failures.append(
                    {
                        "case_id": case_id,
                        "error": f"{type(error).__name__}: {error}",
                        "traceback": traceback.format_exc(),
                    }
                )
            if (
                completed % args.progress_every == 0
                or completed == len(rows)
                or failures
            ):
                elapsed = time.monotonic() - started
                print(
                    "PROGRESS "
                    f"cases={completed}/{len(rows)} created={created} "
                    f"existing={existing} failed={len(failures)} "
                    f"elapsed_s={elapsed:.1f} cases_per_s={completed/max(elapsed,1e-9):.3f}",
                    flush=True,
                )
    failed_path = output / "failed_cases.jsonl"
    with failed_path.open("w", encoding="utf-8") as handle:
        for failure in failures:
            handle.write(json.dumps(failure, sort_keys=True) + "\n")
    if failures:
        raise RuntimeError(
            f"{len(failures)} cases failed; see {failed_path}"
        )

    n_cases, n_organs = _write_manifests(rows, output)
    elapsed = time.monotonic() - started
    summary = {
        "status": "ORGAN_SUV_CDF128_LIBRARY_COMPLETE",
        "cases": n_cases,
        "case_organ_rows": n_organs,
        "created_cases": created,
        "resumed_existing_cases": existing,
        "failed_cases": 0,
        "elapsed_seconds": elapsed,
        "cases_per_second": n_cases / max(elapsed, 1e-9),
        "contract": str(contract_path),
        "case_manifest": str(output / "case_manifest.csv"),
        "organ_manifest": str(output / "organ_manifest.csv"),
    }
    temporary_summary = output / "summary.json.tmp"
    temporary_summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_summary, output / "summary.json")
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
