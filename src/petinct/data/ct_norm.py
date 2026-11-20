"""Canonical CT intensity normalization for MAISI AVAE training.

Geometry is shared with the paired PET pipeline, while intensities follow the
official MAISI CT contract: clip Hounsfield units to ``[-1000, 1000]`` and map
them linearly to ``[0, 1]``.
"""

from __future__ import annotations

import numpy as np


def _validate(hu_min: float, hu_max: float) -> None:
    if not np.isfinite(hu_min) or not np.isfinite(hu_max):
        raise ValueError(f"HU bounds must be finite, got {hu_min}, {hu_max}")
    if hu_min >= hu_max:
        raise ValueError(
            f"HU bounds must satisfy hu_min < hu_max, got {hu_min}, {hu_max}"
        )


def ct_normalize(
    hu,
    hu_min: float = -1000.0,
    hu_max: float = 1000.0,
):
    """Clip CT HU values and map them to ``[0, 1]``."""
    _validate(hu_min, hu_max)
    array = np.asarray(hu, dtype=np.float32)
    clipped = np.clip(array, hu_min, hu_max)
    return (clipped - hu_min) / (hu_max - hu_min)


def ct_denormalize(
    x,
    hu_min: float = -1000.0,
    hu_max: float = 1000.0,
):
    """Invert :func:`ct_normalize` within the configured HU window."""
    _validate(hu_min, hu_max)
    array = np.asarray(x, dtype=np.float32)
    clipped = np.clip(array, 0.0, 1.0)
    return clipped * (hu_max - hu_min) + hu_min
