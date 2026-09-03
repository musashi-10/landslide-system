"""
GIS configuration.

All spatial parameters are centralised here so they can be overridden without
touching algorithm code.  Override via environment variables or by constructing
GISSettings directly in tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class GISSettings(BaseSettings):
    # ------------------------------------------------------------------ #
    # Coordinate reference system
    # ------------------------------------------------------------------ #
    # EPSG:4326 – WGS-84 geographic.
    # All outputs are published in this CRS; internal work may reproject.
    target_crs: str = "EPSG:4326"

    # ------------------------------------------------------------------ #
    # Spatial resolution
    # ------------------------------------------------------------------ #
    # Grid cell side length in *degrees* when target_crs is geographic.
    # ~0.001° ≈ 111 m at the equator.  Keep small for test fixtures.
    grid_resolution_deg: float = Field(default=0.01, gt=0.0)

    # ------------------------------------------------------------------ #
    # Data source paths (override in production via env vars)
    # ------------------------------------------------------------------ #
    dem_path: str = ""          # Path to DEM GeoTIFF
    soil_path: str = ""         # Path to soil vector / raster
    geology_path: str = ""      # Path to geology vector / raster
    landcover_path: str = ""    # Path to land-cover raster
    drainage_path: str = ""     # Path to drainage vector
    historical_path: str = ""   # Path to historical landslide vector

    # ------------------------------------------------------------------ #
    # Output paths
    # ------------------------------------------------------------------ #
    output_dir: str = "gis_outputs"

    # ------------------------------------------------------------------ #
    # Nodata sentinel
    # ------------------------------------------------------------------ #
    nodata_value: float = -9999.0

    # ------------------------------------------------------------------ #
    # Baseline susceptibility weights (must sum to 1.0)
    # WARNING: These weights are NOT scientifically validated.
    # They are a configurable placeholder until Engineer 5's ML model
    # is integrated.  Do NOT use these for operational risk decisions.
    # ------------------------------------------------------------------ #
    susceptibility_weights: Dict[str, float] = {
        "slope":               0.30,
        "elevation":           0.10,
        "historical_landslide": 0.25,
        "land_cover":          0.15,
        "soil_type":           0.10,
        "geology":             0.10,
    }

    # ------------------------------------------------------------------ #
    # Processing version (for metadata / provenance)
    # ------------------------------------------------------------------ #
    processing_version: str = "0.1.0"

    model_config = SettingsConfigDict(
        env_prefix="GIS_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


_settings: GISSettings | None = None


def get_gis_settings() -> GISSettings:
    """Return (and cache) GIS settings."""
    global _settings
    if _settings is None:
        _settings = GISSettings()
    return _settings


def reset_gis_settings() -> None:
    """Reset cached settings (useful in tests)."""
    global _settings
    _settings = None
