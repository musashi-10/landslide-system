"""
satellite/preprocessing/pipeline.py
--------------------------------------
Preprocessing utilities applied to raw satellite band arrays.

Each function operates on ``np.ma.MaskedArray`` objects so that invalid
pixels are propagated correctly through the entire pipeline without
ever silently becoming zeros (honoring the data-contract missing-value
rule).

Preprocessing steps applied
----------------------------
1. ``mask_invalid``  — mask nodata, physically impossible reflectance values.
2. ``normalize_band`` — scale values to the [0, 1] reflectance range.
3. ``apply_cloud_mask`` — exclude cloud-contaminated pixels.
4. ``preprocess_scene`` — convenience wrapper that applies all three steps
                          to every band in a scene dict.

Cloud masking note
------------------
Full Sentinel-2 L2A cloud masking requires the Scene Classification Layer
(SCL band, band 20 in L2A).  For the prototype, ``apply_cloud_mask``
accepts an externally supplied boolean mask (True = cloud pixel).  This
keeps the preprocessing function pure and testable without reading the
real SCL file.  If no cloud mask is supplied the step is a no-op.

CRS note
---------
Preprocessing operates on raw array values only; it does not touch the
spatial reference.  CRS is preserved unchanged from ingestion.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Sentinel-2 L2A reflectance is physically bounded to [0, 1] after scaling.
# Values outside this range indicate sensor noise, atmospheric artefacts, or
# processing errors and should be masked.
REFLECTANCE_MIN: float = 0.0
REFLECTANCE_MAX: float = 1.0


def mask_invalid(
    array: np.ma.MaskedArray,
    low: float = REFLECTANCE_MIN,
    high: float = REFLECTANCE_MAX,
) -> np.ma.MaskedArray:
    """Mask physically impossible reflectance values.

    Pixels already masked (e.g. nodata from :func:`load_raster_band`) are
    left masked.  Additional pixels outside ``[low, high]`` are also masked.

    Parameters
    ----------
    array : np.ma.MaskedArray
        Input band array (float, nominally [0, 1]).
    low : float
        Minimum valid reflectance value.
    high : float
        Maximum valid reflectance value.

    Returns
    -------
    np.ma.MaskedArray
        Copy with out-of-range pixels masked.
    """
    arr = np.ma.array(array, copy=True)
    # Mask NaN / Inf first
    arr = np.ma.masked_invalid(arr)
    # Mask out-of-physical-range values
    arr = np.ma.masked_outside(arr, low, high)
    masked_count = int(np.sum(arr.mask)) if np.ndim(arr.mask) > 0 else 0
    logger.debug("mask_invalid: masked %d pixels", masked_count)
    return arr


def normalize_band(
    array: np.ma.MaskedArray,
    low: float = REFLECTANCE_MIN,
    high: float = REFLECTANCE_MAX,
) -> np.ma.MaskedArray:
    """Scale band values to [0, 1] using a fixed physical range.

    For Sentinel-2 L2A data already divided by the scale factor (10 000),
    this is a pass-through that clips to [0, 1].  For raw DN data that
    has NOT been scaled, set ``low=0, high=10000``.

    The data-contract requirement that missing values are never silently
    zero is preserved: masked pixels remain masked after normalization.

    Parameters
    ----------
    array : np.ma.MaskedArray
        Input band array.
    low : float
        Value that maps to 0.0.
    high : float
        Value that maps to 1.0.

    Returns
    -------
    np.ma.MaskedArray
        Normalized array clipped to [0, 1], same mask as input.
    """
    arr = np.ma.array(array, copy=True, dtype=np.float32)
    if high == low:
        logger.warning("normalize_band: low == high (%s), returning zeros", low)
        arr[:] = 0.0
        return arr

    arr = (arr - low) / (high - low)
    arr = np.ma.clip(arr, 0.0, 1.0).astype(np.float32)
    return arr


def apply_cloud_mask(
    array: np.ma.MaskedArray,
    cloud_mask: np.ndarray | None = None,
) -> np.ma.MaskedArray:
    """Exclude cloud-contaminated pixels from a band array.

    Parameters
    ----------
    array : np.ma.MaskedArray
        Input band array.
    cloud_mask : np.ndarray of bool, or None
        Boolean array where ``True`` means the pixel is cloud-contaminated
        and should be masked.  If ``None``, the array is returned unchanged.

    Returns
    -------
    np.ma.MaskedArray
        Input array with cloud pixels additionally masked.
    """
    if cloud_mask is None:
        return array

    if cloud_mask.shape != array.shape:
        raise ValueError(
            f"cloud_mask shape {cloud_mask.shape} != array shape {array.shape}"
        )

    arr = np.ma.array(array, copy=True)
    # Combine existing mask with cloud mask (OR logic)
    existing_mask = np.ma.getmaskarray(arr)
    arr.mask = existing_mask | cloud_mask.astype(bool)

    cloud_pixel_count = int(np.sum(cloud_mask))
    logger.debug("apply_cloud_mask: masked %d cloud pixels", cloud_pixel_count)
    return arr


def preprocess_scene(
    scene: dict[str, Any],
    cloud_masks: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Apply full preprocessing pipeline to all bands in a scene dict.

    Processing order:
    1. ``mask_invalid`` — physical range filter
    2. ``apply_cloud_mask`` — cloud masking (if cloud_masks supplied)

    Normalization is NOT repeated here because :func:`~satellite.ingestion.loader.load_scene`
    already applies the Sentinel-2 scale factor.

    Parameters
    ----------
    scene : dict
        Scene dict as returned by :func:`~satellite.ingestion.loader.load_scene`
        or :func:`~satellite.ingestion.loader.scene_from_arrays`.
    cloud_masks : dict mapping band name → bool array, or None
        Per-band cloud masks.  If a band is not in the dict, no cloud mask
        is applied for that band.

    Returns
    -------
    scene : dict
        New scene dict with preprocessed band arrays.  ``transform`` and
        ``crs`` keys are preserved unchanged.
    """
    cloud_masks = cloud_masks or {}
    processed_bands: dict[str, np.ma.MaskedArray] = {}

    for band_name, arr in scene["bands"].items():
        arr = mask_invalid(arr)
        arr = apply_cloud_mask(arr, cloud_masks.get(band_name))
        processed_bands[band_name] = arr

    return {
        "bands": processed_bands,
        "transform": scene["transform"],
        "crs": scene["crs"],
        "scene_dir": scene.get("scene_dir", "unknown"),
    }
