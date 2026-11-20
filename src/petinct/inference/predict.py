"""Generate PET in SUV units from a CT and its aligned anatomical labels."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np

from petinct.inference.generate_ctpet_coarse_cache import _atomic_nifti, file_sha256


def run_module(name: str, *arguments: Path | str) -> None:
    subprocess.run(
        [sys.executable, "-m", name, *(str(value) for value in arguments)],
        check=True,
    )


def predict(
    ct: Path, anatomy: Path, model_dir: Path, output: Path, work: Path
) -> None:
    anatomy_map = model_dir / "anatomy_labels.csv"
    config = model_dir / "generator.json"
    worm = model_dir / "worm.pt"
    body = model_dir / "body.pt"
    brain = model_dir / "brain.pt"
    manifest = work / "input.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["case_id", "patient_group_hash", "split", "ct_path", "anatomy_path", "pet_path"]
        )
        writer.writerow(["input", "input", "demo", ct, anatomy, ""])

    ct_library = work / "ct_distributions"
    run_module(
        "petinct.data.build_organ_ct_distribution_library",
        "--paired-manifest", manifest, "--anatomy-label-map", anatomy_map,
        "--out-dir", ct_library, "--splits", "demo", "--workers", "1",
    )
    condition = work / "condition"
    run_module(
        "petinct.data.build_worm_pet_probability_cache",
        "--ct-library-root", ct_library, "--checkpoint", worm,
        "--checkpoint-sha256", file_sha256(worm), "--output-dir", condition,
        "--splits", "demo", "--batch-size", "1",
    )
    for region, checkpoint in (("body", body), ("brain", brain)):
        arguments = []
        if region == "brain":
            arguments = [
                "--crop-target-host", "brain", "--mask-output-to-crop-target-host",
                "--skip-missing-crop-target-host",
            ]
        run_module(
            "petinct.inference.generate_ctpet_coarse_cache",
            "--checkpoint", checkpoint, "--config", config,
            "--paired-manifest", manifest, "--anatomy-label-map-csv", anatomy_map,
            "--distribution-condition-root", condition / "predicted_probability",
            "--out-dir", work / (region + "_run"), "--prediction-root", work / region,
            "--split", "demo", "--limit-cases", "1", "--expected-total-cases", "1",
            "--expected-spacing", "1.5,1.5,3.0", *arguments,
        )

    image = nib.load(work / "body" / "input.nii.gz")
    pet = image.get_fdata(dtype=np.float32)
    brain_path = work / "brain" / "input.nii.gz"
    if brain_path.is_file():
        with anatomy_map.open(newline="", encoding="utf-8") as stream:
            brain_ids = [
                int(row["label_id"])
                for row in csv.DictReader(stream)
                if "brain" in (row["target_host"].strip().lower(), row["label_name"].strip().lower())
            ]
        labels = np.asanyarray(nib.load(anatomy).dataobj)
        brain_mask = np.isin(labels, brain_ids)
        brain_pet = nib.load(brain_path).get_fdata(dtype=np.float32)
        pet[brain_mask] = brain_pet[brain_mask]
    _atomic_nifti(output, pet, image)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ct", type=Path, required=True, help="CT in Hounsfield units (NIfTI)")
    parser.add_argument("--anatomy", type=Path, required=True, help="Aligned anatomical labels (NIfTI)")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Output PET volume (.nii.gz)")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.name.endswith(".nii.gz"):
        parser.error("--output must end in .nii.gz")
    if output.exists():
        parser.error(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="petinct-") as directory:
        predict(
            args.ct.resolve(), args.anatomy.resolve(), args.model_dir.resolve(),
            output, Path(directory),
        )


if __name__ == "__main__":
    main()
