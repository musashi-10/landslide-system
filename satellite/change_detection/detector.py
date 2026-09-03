"""
satellite/change_detection/detector.py
----------------------------------------
Temporal change-detection utilities for satellite imagery.

Method: NDVI-delta (bi-temporal differencing)
----------------------------------------------
This is the simplest and most interpretable change-detection approach:

    ΔNDVI = NDVI(t2) − NDVI(t1)

    Negative ΔNDVI → vegetation loss (possible disturbance / landslide scar)
    Positive ΔNDVI → vegetation gain (revegetation)
    ~Zero ΔNDVI   → stable surface

Why NDVI-delta?
---------------
* No labeled data required.
* Directly interpretable.
* Widely validated in landslide literature
  (e.g. Lin et al. 2004; Mondini et al. 2011).
* Computational cost: one array subtraction.

⚠ Critical limitation
----------------------
A large negative ΔNDVI is a change *indicator*, NOT a landslide *label*.
Causes of vegetation loss include:
  - Seasonal change (senescence)
  - Agricultural clearing
  - Fire
  - Landslide (the signal we want)
  - Cloud/shadow contamination

The ``detect_disturbance`` function therefore outputs a continuous
``ndvi_change`` value and a binary ``disturbance_mask`` for thresholded
flagging.  Both are provided as *features* to the GIS/ML pipeline, not
as definitive landslide detections.

For production: combine ΔNDVI with slope, soil type, and rainfall to
improve signal-to-noise ratio (this is the ML engineer's job).

Image pair requirements
-----------------------
* Both images must be spatially co-registered (same grid, same CRS).
* Seasonal effects should be minimised by comparing same-season images
  or by applying a seasonal baseline anomaly.
* Cloud-masked pixels in *either* image should be excluded from the delta.

CRS / temporal note
-------------------
* This module operates on array values; CRS is unchanged.
* Acquisition times of BOTH images must be recorded in the output
  provenance (t1_time, t2_time) so the ML engineer can compute the
  time delta and avoid leakage across event dates.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def compute_ndvi_change(
    ndvi_t1: np.ma.MaskedArray,
    ndvi_t2: np.ma.MaskedArray,
) -> np.ma.MaskedArray:
    """Compute per-pixel NDVI change between two acquisitions.

    ΔNDVI = NDVI(t2) − NDVI(t1)

    Both arrays must have the same shape.  Pixels that are masked in
    *either* image are masked in the output (conservative approach — we
    do not estimate change where we have no data).

    Parameters
    ----------
    ndvi_t1 : np.ma.MaskedArray
        NDVI at the earlier time (t1).  Range [−1, 1].
    ndvi_t2 : np.ma.MaskedArray
        NDVI at the later time (t2).  Range [−1, 1].

    Returns
    -------
    ndvi_change : np.ma.MaskedArray
        Per-pixel NDVI delta.  Range (−2, 2) theoretically; practically
        within (−1, 1).  Masked where either input was masked.
    """
    if ndvi_t1.shape != ndvi_t2.shape:
        raise ValueError(
            f"ndvi_t1 shape {ndvi_t1.shape} != ndvi_t2 shape {ndvi_t2.shape}"
        )

    t1 = np.ma.array(ndvi_t1, dtype=np.float32, copy=True)
    t2 = np.ma.array(ndvi_t2, dtype=np.float32, copy=True)

    delta = t2 - t1

    # Mask pixels that were masked in either image
    combined_mask = np.ma.getmaskarray(t1) | np.ma.getmaskarray(t2)
    delta.mask = combined_mask

    logger.debug(
        "compute_ndvi_change: valid pixels=%d  mean_delta=%.3f  min=%.3f  max=%.3f",
        int(delta.count()),
        float(delta.mean()) if delta.count() > 0 else float("nan"),
        float(delta.min()) if delta.count() > 0 else float("nan"),
        float(delta.max()) if delta.count() > 0 else float("nan"),
    )
    return delta


def detect_disturbance(
    ndvi_change: np.ma.MaskedArray,
    threshold: float = -0.15,
) -> dict[str, Any]:
    """Flag pixels with significant vegetation loss as potential disturbances.

    A pixel is flagged when ΔNDVI ≤ ``threshold`` (i.e., notable vegetation
    decrease).  The default threshold of −0.15 is a commonly cited value in
    the change-detection literature for detecting abrupt disturbances.

    ⚠ The output disturbance mask is a *feature*, not a landslide label.

    Parameters
    ----------
    ndvi_change : np.ma.MaskedArray
        NDVI delta array from :func:`compute_ndvi_change`.
    threshold : float
        Vegetation-loss threshold.  Should be negative.  Pixels with
        ΔNDVI ≤ threshold are flagged.  Default: −0.15.

    Returns
    -------
    dict with keys
        ``"disturbance_mask"``    → np.ndarray of bool (True = flagged)
        ``"ndvi_change"``         → np.ma.MaskedArray (unchanged input)
        ``"threshold_used"``      → float
        ``"flagged_pixel_count"`` → int
        ``"total_valid_pixels"``  → int
        ``"flagged_fraction"``    → float or None
    """
    if threshold >= 0.0:
        logger.warning(
            "detect_disturbance: threshold=%.2f is non-negative.  "
            "Vegetation-loss thresholds should be negative.",
            threshold,
        )

    # Flagged where change is <= threshold AND pixel is valid
    valid = ~np.ma.getmaskarray(ndvi_change)
    data = np.ma.filled(ndvi_change, np.nan)

    disturbance_mask = valid & (data <= threshold)
    flagged_count = int(np.sum(disturbance_mask))
    total_valid = int(np.sum(valid))
    flagged_fraction = flagged_count / total_valid if total_valid > 0 else None

    logger.info(
        "detect_disturbance: threshold=%.2f  flagged=%d / %d valid pixels (%.1f%%)",
        threshold,
        flagged_count,
        total_valid,
        (flagged_fraction * 100) if flagged_fraction is not None else 0.0,
    )

    return {
        "disturbance_mask": disturbance_mask,
        "ndvi_change": ndvi_change,
        "threshold_used": threshold,
        "flagged_pixel_count": flagged_count,
        "total_valid_pixels": total_valid,
        "flagged_fraction": flagged_fraction,
    }
