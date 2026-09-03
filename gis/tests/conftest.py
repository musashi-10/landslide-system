"""
GIS test configuration.

Skip all GIS tests when the heavy geospatial dependencies (rasterio, geopandas)
are not installed.  This keeps ``pytest -q`` green on machines that only have
the ML / core stack installed.
"""

import pytest


def _geospatial_available() -> bool:
    try:
        import rasterio  # noqa: F401
        import geopandas  # noqa: F401
        return True
    except ImportError:
        return False


_SKIP_REASON = "rasterio / geopandas not installed — skipping GIS tests"


def pytest_ignore_collect(collection_path, config):
    """Tell pytest to skip collecting test_gis.py when deps are absent."""
    if not _geospatial_available() and "test_gis" in collection_path.name:
        return True
    return None
