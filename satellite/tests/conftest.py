"""
satellite/tests/conftest.py
----------------------------
Shared pytest fixtures for the satellite test suite.

All fixtures produce synthetic 10×10 NumPy arrays.
No real satellite imagery or internet access is required.

Coordinate system used in fixtures
------------------------------------
CRS      : EPSG:4326 (WGS-84 geographic)
Origin   : top-left pixel at (lon=88.0, lat=27.01)
Pixel    : 0.0001° × 0.0001° ≈ 11 m at this latitude
Shape    : (10 rows, 10 cols)

The affine transform is built to match the rasterio convention:
    | a  b  c |   |  pixel_width  0              origin_x |
    | d  e  f | = |  0            -pixel_height  origin_y |
    | 0  0  1 |

where origin_x = left edge longitude, origin_y = top edge latitude.
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FIXTURE_ROWS = 10
FIXTURE_COLS = 10
PIXEL_SIZE_DEG = 0.0001  # degrees per pixel
ORIGIN_LON = 88.0
ORIGIN_LAT = 27.01  # top-left corner latitude (north)
ACQUISITION_TIME = "2024-06-15T10:23:00Z"
SOURCE_CRS = "EPSG:4326"


# ---------------------------------------------------------------------------
# Affine transform fixture
# ---------------------------------------------------------------------------
class SimpleTransform:
    """Minimal affine transform compatible with the alignment module.

    Mirrors the structure of ``affine.Affine`` without requiring the package.
    ``a`` = pixel width (longitude step),  ``e`` = pixel height (negative for N-up).
    ``c`` = origin_x (left edge lon),      ``f`` = origin_y (top edge lat).
    ``b``, ``d`` = 0.0 (no rotation).
    """

    def __init__(
        self,
        pixel_width: float,
        pixel_height: float,
        origin_x: float,
        origin_y: float,
    ) -> None:
        self.a = pixel_width
        self.b = 0.0
        self.c = origin_x
        self.d = 0.0
        self.e = -pixel_height  # negative: top-down raster
        self.f = origin_y


@pytest.fixture
def affine_transform() -> SimpleTransform:
    """Return a simple affine transform for the 10×10 fixture grid."""
    return SimpleTransform(
        pixel_width=PIXEL_SIZE_DEG,
        pixel_height=PIXEL_SIZE_DEG,
        origin_x=ORIGIN_LON,
        origin_y=ORIGIN_LAT,
    )


# ---------------------------------------------------------------------------
# Band array fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def red_band() -> np.ma.MaskedArray:
    """10×10 synthetic Red band (Sentinel-2 B04 equivalent).

    Values drawn from uniform [0.05, 0.25] — typical bare-to-vegetated
    red reflectance.  All pixels valid.
    """
    rng = np.random.default_rng(seed=42)
    data = rng.uniform(0.05, 0.25, size=(FIXTURE_ROWS, FIXTURE_COLS)).astype(np.float32)
    return np.ma.MaskedArray(data)


@pytest.fixture
def nir_band() -> np.ma.MaskedArray:
    """10×10 synthetic NIR band (Sentinel-2 B08 equivalent).

    Values drawn from uniform [0.20, 0.50] — typical NIR reflectance
    range for vegetated and bare surfaces.  All pixels valid.
    """
    rng = np.random.default_rng(seed=43)
    data = rng.uniform(0.20, 0.50, size=(FIXTURE_ROWS, FIXTURE_COLS)).astype(np.float32)
    return np.ma.MaskedArray(data)


@pytest.fixture
def swir_band() -> np.ma.MaskedArray:
    """10×10 synthetic SWIR band (Sentinel-2 B11 equivalent).

    Values drawn from uniform [0.05, 0.30] — typical SWIR range.
    All pixels valid.
    """
    rng = np.random.default_rng(seed=44)
    data = rng.uniform(0.05, 0.30, size=(FIXTURE_ROWS, FIXTURE_COLS)).astype(np.float32)
    return np.ma.MaskedArray(data)


@pytest.fixture
def red_band_with_masked() -> np.ma.MaskedArray:
    """10×10 Red band with 4 masked (invalid) pixels in the top-left corner."""
    rng = np.random.default_rng(seed=100)
    data = rng.uniform(0.05, 0.25, size=(FIXTURE_ROWS, FIXTURE_COLS)).astype(np.float32)
    mask = np.zeros((FIXTURE_ROWS, FIXTURE_COLS), dtype=bool)
    mask[0:2, 0:2] = True  # mask top-left 2×2
    return np.ma.MaskedArray(data, mask=mask)


@pytest.fixture
def vegetation_ndvi() -> np.ma.MaskedArray:
    """Uniform NDVI array representing dense forest (~0.7).

    Used for testing land cover classification (should → forest).
    """
    data = np.full((FIXTURE_ROWS, FIXTURE_COLS), 0.70, dtype=np.float32)
    return np.ma.MaskedArray(data)


@pytest.fixture
def bare_ndvi() -> np.ma.MaskedArray:
    """Uniform NDVI array representing bare ground (~0.05).

    Used for testing land cover classification (should → bare with high BSI).
    """
    data = np.full((FIXTURE_ROWS, FIXTURE_COLS), 0.05, dtype=np.float32)
    return np.ma.MaskedArray(data)


@pytest.fixture
def high_bsi() -> np.ma.MaskedArray:
    """Uniform BSI array representing strongly bare/exposed surface (0.3)."""
    data = np.full((FIXTURE_ROWS, FIXTURE_COLS), 0.30, dtype=np.float32)
    return np.ma.MaskedArray(data)


@pytest.fixture
def zero_denominator_red() -> np.ma.MaskedArray:
    """Red band all zeros — forces NDVI denominator to zero for NIR=0."""
    data = np.zeros((FIXTURE_ROWS, FIXTURE_COLS), dtype=np.float32)
    return np.ma.MaskedArray(data)


@pytest.fixture
def zero_nir() -> np.ma.MaskedArray:
    """NIR band all zeros — combined with zero_denominator_red → NDVI denom=0."""
    data = np.zeros((FIXTURE_ROWS, FIXTURE_COLS), dtype=np.float32)
    return np.ma.MaskedArray(data)


# ---------------------------------------------------------------------------
# Scene dict fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_scene(red_band, nir_band, swir_band, affine_transform) -> dict:
    """Full synthetic scene dict with red, nir, swir bands."""
    return {
        "bands": {
            "red": red_band,
            "nir": nir_band,
            "swir": swir_band,
        },
        "transform": affine_transform,
        "crs": SOURCE_CRS,
        "scene_dir": "synthetic",
    }
