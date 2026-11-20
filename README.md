# PETinCT

Generate a whole-body FDG PET prediction from CT.

## Run the example

Use Python 3.10 or later and a CUDA-enabled NVIDIA GPU. Download `petinct-models.tar.gz` and `petinct-example.tar.gz` from the [demo release](../../releases/tag/demo) into the repository directory.

```sh
pip install .
tar -xzf petinct-models.tar.gz
tar -xzf petinct-example.tar.gz
petinct --ct example/ct.nii.gz --anatomy example/anatomy.nii.gz \
  --model-dir models --output output/predicted_pet.nii.gz
```

Open the CT and the predicted PET together in a NIfTI viewer such as 3D Slicer. The PET output is on the CT grid in standardized uptake value (SUV) units.

The example includes the anatomical label map used by the model. For another CT, provide the corresponding fused anatomical label map on the same 1.5 × 1.5 × 3 mm grid.

The example CT comes from Gatidis and Küstner, [FDG-PET-CT-Lesions](https://doi.org/10.7937/gkr0-xv29), via the [NCI Imaging Data Commons](https://portal.imaging.datacommons.cancer.gov/collections/fdg_pet_ct_lesions/). The defaced CT was converted to NIfTI and resampled under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
