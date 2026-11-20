# PETinCT

Generate a whole-body FDG PET prediction from CT.

## Run the example

Use Python 3.10 or later and a CUDA-enabled NVIDIA GPU. Download and extract the models and example from Releases.

```sh
pip install .
petinct --ct example/ct.nii.gz --anatomy example/anatomy.nii.gz \
  --model-dir models --output output/predicted_pet.nii.gz
```

Open the CT and the predicted PET together in a NIfTI viewer such as 3D Slicer. The PET output is on the CT grid in standardized uptake value (SUV) units.

The example includes the anatomical label map used by the model. For another CT, provide the corresponding fused anatomical label map on the same 1.5 × 1.5 × 3 mm grid.
