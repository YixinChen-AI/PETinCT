"""Build an aligned full-volume patient-by-organ CT HU CDF library.

The case, anatomy and archive schema intentionally mirrors the canonical PET
SUV library.  CT and anatomy must already share the formal
1.5 x 1.5 x 3.0 mm grid.  Background is excluded and every present foreground
organ is represented by an unsmoothed 128-bin empirical CDF over the fixed
physical HU support [-1000, 1000].
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from petinct.data.build_organ_suv_distribution_library import (
    _atomic_save_npz,
    _counts_to_cdf,
    _load_label_map,
    _safe_case_name,
)


REQUIRED_MANIFEST_COLUMNS = {
    "case_id",
    "patient_group_hash",
    "ct_path",
    "anatomy_path",
    "split",
}


@dataclass(frozen=True)
class CtDistributionContract:
    bins: int
    hu_min: float
    hu_max: float
    expected_spacing: tuple[float, float, float]
    affine_tolerance: float = 1e-3
    spacing_tolerance: float = 1e-3

    @property
    def bin_width(self) -> float:
        return (self.hu_max - self.hu_min) / self.bins

    @property
    def bin_edges(self) -> np.ndarray:
        return np.linspace(
            self.hu_min,
            self.hu_max,
            self.bins + 1,
            dtype=np.float64,
        )


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
            "clip_min_fraction",
            "clip_max_fraction",
            "cdf_128",
            "spacing",
        }
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(
                f"incomplete existing CT archive {path}: {sorted(missing)}"
            )
        return {key: archive[key] for key in required}


def _process_case(
    row: dict[str, str],
    output_dir: str,
    label_names: dict[int, str],
    model_label_ids: dict[int, int],
    contract: CtDistributionContract,
) -> dict[str, Any]:
    import nibabel as nib

    case_id = row["case_id"]
    case_path = Path(output_dir) / "cases" / _safe_case_name(case_id)
    if case_path.exists():
        archive = _read_existing(case_path)
        if (
            str(archive["case_id"].item()) != case_id
            or str(archive["patient_group_hash"].item())
            != row["patient_group_hash"]
            or str(archive["split"].item()) != row["split"]
        ):
            raise ValueError(f"existing CT archive identity mismatch: {case_path}")
        return {
            "status": "existing",
            "case_id": case_id,
            "n_organs": int(archive["label_ids"].shape[0]),
        }

    ct_image = nib.load(row["ct_path"])
    anatomy_image = nib.load(row["anatomy_path"])
    if ct_image.shape != anatomy_image.shape or len(ct_image.shape) != 3:
        raise ValueError(
            f"CT/anatomy shape mismatch for {case_id}: "
            f"{ct_image.shape} vs {anatomy_image.shape}"
        )
    if not np.allclose(
        ct_image.affine,
        anatomy_image.affine,
        atol=contract.affine_tolerance,
        rtol=0.0,
    ):
        raise ValueError(f"CT/anatomy affine mismatch for {case_id}")
    spacing = tuple(float(value) for value in ct_image.header.get_zooms()[:3])
    if not np.allclose(
        spacing,
        contract.expected_spacing,
        atol=contract.spacing_tolerance,
        rtol=0.0,
    ):
        raise ValueError(
            f"spacing contract violation for {case_id}: {spacing}"
        )

    ct = np.asanyarray(ct_image.dataobj, dtype=np.float32)
    anatomy_float = np.asanyarray(anatomy_image.dataobj)
    if not np.isfinite(ct).all():
        raise ValueError(f"non-finite CT for {case_id}")
    rounded = np.rint(anatomy_float)
    if not np.array_equal(anatomy_float, rounded):
        raise ValueError(f"non-integer anatomy labels for {case_id}")
    anatomy = rounded.astype(np.int64, copy=False)
    if anatomy.min(initial=0) < 0:
        raise ValueError(f"negative anatomy labels for {case_id}")

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

    values = ct[foreground].astype(np.float64, copy=False)
    clipped = np.clip(values, contract.hu_min, contract.hu_max)
    bin_indices = np.floor(
        (clipped - contract.hu_min) / contract.bin_width
    ).astype(np.int64)
    np.clip(bin_indices, 0, contract.bins - 1, out=bin_indices)

    maximum_label = int(observed.max())
    joint = labels * contract.bins + bin_indices
    counts = np.bincount(
        joint,
        minlength=(maximum_label + 1) * contract.bins,
    ).reshape(maximum_label + 1, contract.bins)
    voxel_counts = np.bincount(labels, minlength=maximum_label + 1)
    minimum_counts = np.bincount(
        labels[values <= contract.hu_min],
        minlength=maximum_label + 1,
    )
    maximum_counts = np.bincount(
        labels[values >= contract.hu_max],
        minlength=maximum_label + 1,
    )
    cdf = np.stack(
        [_counts_to_cdf(counts[int(label_id)]) for label_id in observed]
    )
    selected_counts = voxel_counts[observed].astype(np.int64)
    if np.any(selected_counts <= 0):
        raise ValueError(f"non-positive organ voxel count for {case_id}")

    label_ids = observed.astype(np.int32)
    voxel_volume_mm3 = float(np.prod(spacing))
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
        clip_min_fraction=(
            minimum_counts[observed] / selected_counts
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
        expected = by_case[case_id]
        if (
            str(archive["patient_group_hash"].item())
            != expected["patient_group_hash"]
            or str(archive["split"].item()) != expected["split"]
        ):
            raise ValueError(f"CT archive/manifest mismatch for {case_id}")
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
                    "clip_min_fraction": float(
                        archive["clip_min_fraction"][index]
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
            f"CT case archive count mismatch: {len(case_rows)} vs {len(rows)}"
        )
    tables = (
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
                "clip_min_fraction",
                "clip_max_fraction",
                "distribution_path",
                "distribution_row",
            ],
            organ_rows,
        ),
    )
    for path, fieldnames, contents in tables:
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
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--bins", type=int, default=128)
    parser.add_argument("--hu-min", type=float, default=-1000.0)
    parser.add_argument("--hu-max", type=float, default=1000.0)
    parser.add_argument("--expected-spacing", default="1.5,1.5,3.0")
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--allow-extend-completed", action="store_true")
    args = parser.parse_args()

    if args.bins != 128:
        raise ValueError("canonical organ CT CDF requires exactly 128 bins")
    if args.hu_min != -1000.0 or args.hu_max != 1000.0:
        raise ValueError("canonical CT support must be [-1000,1000] HU")
    if args.workers < 1 or args.progress_every < 1:
        raise ValueError("workers and progress-every must be positive")
    expected_spacing = tuple(
        float(value) for value in args.expected_spacing.split(",")
    )
    if len(expected_spacing) != 3:
        raise ValueError("expected-spacing must contain three values")
    contract = CtDistributionContract(
        bins=args.bins,
        hu_min=args.hu_min,
        hu_max=args.hu_max,
        expected_spacing=expected_spacing,
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
                f"completed CT distribution library already exists: {output}"
            )
        history = output / "history"
        history.mkdir(exist_ok=True)
        archived = history / (
            f"summary_before_extension_{time.strftime('%Y%m%d_%H%M%S')}.json"
        )
        os.replace(previous_summary, archived)

    contract_payload = {
        "status": "ORGAN_CT_CDF128_CONTRACT",
        "paired_manifest": str(args.paired_manifest.resolve()),
        "anatomy_label_map": str(args.anatomy_label_map.resolve()),
        "splits": sorted(splits),
        "selected_cases": len(rows),
        "background_label": 0,
        "background_included": False,
        "label_policy": "dynamic_present_foreground_subset_of_canonical_map",
        "full_volume": True,
        "target": "physical CT Hounsfield units",
        "hu_clip": [contract.hu_min, contract.hu_max],
        "bins": contract.bins,
        "bin_edges": contract.bin_edges.tolist(),
        "bin_width_hu": contract.bin_width,
        "smoothing": "none",
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
    completed = created = existing = 0
    failures: list[dict[str, str]] = []
    print(
        "ORGAN_CT_CDF_START "
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
                    f"elapsed_s={elapsed:.1f} "
                    f"cases_per_s={completed/max(elapsed,1e-9):.3f}",
                    flush=True,
                )

    failed_path = output / "failed_cases.jsonl"
    with failed_path.open("w", encoding="utf-8") as handle:
        for failure in failures:
            handle.write(json.dumps(failure, sort_keys=True) + "\n")
    if failures:
        raise RuntimeError(
            f"{len(failures)} CT cases failed; see {failed_path}"
        )

    n_cases, n_organs = _write_manifests(rows, output)
    elapsed = time.monotonic() - started
    summary = {
        "status": "ORGAN_CT_CDF128_LIBRARY_COMPLETE",
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
    temporary = output / "summary.json.tmp"
    temporary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output / "summary.json")
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
