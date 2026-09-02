"""
satellite/features/indices.py
-------------------------------
Spectral index calculations and rule-based land-cover classification.

Feature definitions
-------------------

NDVI — Normalized Difference Vegetation Index
    Formula  : (NIR − Red) / (NIR + Red)
    Range    : [−1, 1]
    Source   : Sentinel-2 bands B08 (NIR, 10 m) and B04 (Red, 10 m)
    Why      : Dense vegetation stabilizes slopes.  Low NDVI indicates
               sparse or absent vegetation, increasing susceptibility.
    ML use   : Continuous feature in landslide susceptibility model.

BSI — Bare Soil Index
    Formula  : (SWIR + Red − NIR) / (SWIR + Red + NIR)
    Range    : [−1, 1]  (higher = more bare/exposed soil)
    Source   : Sentinel-2 B11 (SWIR, 20 m resampled to 10 m), B08, B04
    Why      : Bare/exposed slopes have higher erosion risk and lower
               slope stability.  BSI complements NDVI.
    ML use   : Continuous feature; high BSI + steep slope is a risk signal.

Land cover (rule-based)
    Categories : forest, shrub_grass, bare, built, water, unknown
    Method     : Threshold rules on NDVI and BSI.  NOT a trained classifier.
                 This is a prototype-level approximation.  For production,
                 replace with a validated land-cover product (e.g. ESA
                 WorldCover, Google Dynamic World).
    Why        : Land cover is a standard static vulnerability feature
                 listed in the data contract (Section 4).
    ML use     : Categorical feature; also joins with GIS static data.

Scientific note
---------------
These indices are well-established in the remote-sensing literature:
- Tucker (1979): NDVI definition.
- Rikimaru et al. (2002): BSI for forest monitoring.
- Xu (2006): Modified Normalized Difference Water Index used to inform
  water class thresholds.

All calculations preserve masked arrays so that cloud-contaminated or
invalid pixels are never silently turned into valid feature values.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Land-cover classification thresholds
# ---------------------------------------------------------------------------
# These are approximate values; production systems should use validated maps.
_NDVI_WATER_MAX: float = -0.05      # very low / negative NDVI → water
_NDVI_BARE_MAX: float = 0.20        # low NDVI + high BSI → bare
_BSI_BARE_MIN: float = 0.0          # positive BSI supports "bare" label
_NDVI_SHRUB_MAX: float = 0.40       # moderate NDVI → shrub/grassland
_NDVI_FOREST_MIN: float = 0.40      # high NDVI → forest


def compute_ndvi(
    red: np.ma.MaskedArray,
    nir: np.ma.MaskedArray,
) -> np.ma.MaskedArray:
    """Compute the Normalized Difference Vegetation Index.

    NDVI = (NIR − Red) / (NIR + Red)

    Division by zero is handled by masking pixels where NIR + Red == 0.
    Resulting values are clipped to [−1, 1] to handle floating-point noise.

    Parameters
    ----------
    red : np.ma.MaskedArray
        Red band (Sentinel-2 B04), surface reflectance [0, 1].
    nir : np.ma.MaskedArray
        Near-infrared band (Sentinel-2 B08), surface reflectance [0, 1].

    Returns
    -------
    ndvi : np.ma.MaskedArray
        NDVI in [−1, 1].  Pixels with zero denominator are masked.
    """
    red = np.ma.array(red, dtype=np.float32)
    nir = np.ma.array(nir, dtype=np.float32)

    denominator = nir + red
    # Mask zero denominator to avoid division-by-zero
    denominator = np.ma.masked_where(denominator == 0.0, denominator)

    ndvi = (nir - red) / denominator
    ndvi = np.ma.clip(ndvi, -1.0, 1.0)

    logger.debug(
        "compute_ndvi: valid pixels=%d  mean=%.3f",
        int(ndvi.count()),
        float(ndvi.mean()) if ndvi.count() > 0 else float("nan"),
    )
    return ndvi


def compute_bsi(
    red: np.ma.MaskedArray,
    nir: np.ma.MaskedArray,
    swir: np.ma.MaskedArray,
) -> np.ma.MaskedArray:
    """Compute the Bare Soil Index.

    BSI = (SWIR + Red − NIR) / (SWIR + Red + NIR)

    Parameters
    ----------
    red : np.ma.MaskedArray
        Red band (Sentinel-2 B04), surface reflectance [0, 1].
    nir : np.ma.MaskedArray
        NIR band (Sentinel-2 B08), surface reflectance [0, 1].
    swir : np.ma.MaskedArray
        SWIR band (Sentinel-2 B11, resampled to match B04/B08 resolution).

    Returns
    -------
    bsi : np.ma.MaskedArray
        BSI in [−1, 1].  Pixels with zero denominator are masked.
    """
    red = np.ma.array(red, dtype=np.float32)
    nir = np.ma.array(nir, dtype=np.float32)
    swir = np.ma.array(swir, dtype=np.float32)

    numerator = swir + red - nir
    denominator = swir + red + nir
    denominator = np.ma.masked_where(denominator == 0.0, denominator)

    bsi = numerator / denominator
    bsi = np.ma.clip(bsi, -1.0, 1.0)

    logger.debug(
        "compute_bsi: valid pixels=%d  mean=%.3f",
        int(bsi.count()),
        float(bsi.mean()) if bsi.count() > 0 else float("nan"),
    )
    return bsi


def classify_land_cover(
    ndvi: np.ma.MaskedArray,
    bsi: np.ma.MaskedArray,
) -> np.ndarray:
    """Classify pixels into land-cover categories using spectral thresholds.

    This is a rule-based prototype method, NOT a trained classifier.
    For production use, replace with a validated satellite land-cover
    product (e.g. ESA WorldCover 10 m, Google Dynamic World).

    Classification rules (applied in priority order)
    -------------------------------------------------
    1. NDVI < −0.05                          → water
    2. NDVI < 0.20 AND BSI > 0.0            → bare
    3. NDVI < 0.40                           → shrub_grass
    4. NDVI ≥ 0.40                           → forest
    5. Masked / invalid                      → unknown

    Parameters
    ----------
    ndvi : np.ma.MaskedArray
        NDVI array [−1, 1].
    bsi : np.ma.MaskedArray
        BSI array [−1, 1].

    Returns
    -------
    classes : np.ndarray of str
        2-D array of land-cover labels, same shape as input.
    """
    if ndvi.shape != bsi.shape:
        raise ValueError(
            f"ndvi shape {ndvi.shape} != bsi shape {bsi.shape}"
        )

    classes = np.full(ndvi.shape, "unknown", dtype=object)

    # Combined validity mask (True where pixel is usable)
    valid = ~(np.ma.getmaskarray(ndvi) | np.ma.getmaskarray(bsi))
    ndvi_data = np.ma.filled(ndvi, np.nan)
    bsi_data = np.ma.filled(bsi, np.nan)

    # Priority 4: forest (applied first so lower priorities overwrite)
    classes[valid & (ndvi_data >= _NDVI_FOREST_MIN)] = "forest"
    # Priority 3: shrub/grassland
    classes[valid & (ndvi_data >= 0.0) & (ndvi_data < _NDVI_SHRUB_MAX)] = "shrub_grass"
    # Priority 2: bare
    classes[valid & (ndvi_data < _NDVI_BARE_MAX) & (bsi_data > _BSI_BARE_MIN)] = "bare"
    # Priority 1: water (highest priority — overwrites others)
    classes[valid & (ndvi_data < _NDVI_WATER_MAX)] = "water"

    unique, counts = np.unique(classes, return_counts=True)
    logger.debug(
        "classify_land_cover: %s",
        dict(zip(unique.tolist(), counts.tolist())),
    )
    return classes
