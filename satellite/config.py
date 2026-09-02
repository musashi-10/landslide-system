"""
satellite/config.py
--------------------
Configuration object for the satellite feature pipeline.

All tuneable parameters live here.  No hard-coded paths, keys, or
coordinates elsewhere in the package.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class SatelliteConfig:
    """Holds all runtime parameters for the satellite pipeline.

    Parameters
    ----------
    bbox : tuple[float, float, float, float]
        Bounding box as (min_lon, min_lat, max_lon, max_lat) in EPSG:4326.
    start_date : str
        Start of the acquisition window, ISO 8601 date string (YYYY-MM-DD).
    end_date : str
        End of the acquisition window, ISO 8601 date string (YYYY-MM-DD).
    source : str
        Human-readable satellite source label (default ``"Sentinel-2"``).
    spatial_resolution_m : int
        Target output spatial resolution in metres (default ``10``).
    cloud_threshold : float
        Maximum acceptable cloud cover fraction [0.0, 1.0] (default ``0.3``).
    output_crs : str
        EPSG code string for outputs (default ``"EPSG:4326"``).
    cache_dir : str
        Directory for caching downloaded raster data.
    processing_version : str
        Version tag stamped on every output record for reproducibility.
    ndvi_change_threshold : float
        Minimum |ΔNDVI| to flag a pixel as a detected disturbance
        (default ``-0.15``; negative because vegetation *loss* matters).
    """

    # Geographic scope
    bbox: tuple[float, float, float, float] = (88.0, 27.0, 89.0, 28.0)
    start_date: str = "2024-01-01"
    end_date: str = "2024-12-31"

    # Source metadata
    source: str = "Sentinel-2"
    spatial_resolution_m: int = 10

    # Quality filtering
    cloud_threshold: float = 0.30

    # Spatial reference
    output_crs: str = "EPSG:4326"

    # I/O
    cache_dir: str = field(
        default_factory=lambda: os.environ.get(
            "SATELLITE_CACHE_DIR", ".satellite_cache"
        )
    )

    # Provenance
    processing_version: str = field(
        default_factory=lambda: os.environ.get(
            "SATELLITE_PROCESSING_VERSION", "v1"
        )
    )

    # Change detection
    ndvi_change_threshold: float = -0.15

    @classmethod
    def from_env(cls) -> "SatelliteConfig":
        """Construct a ``SatelliteConfig`` from environment variables.

        Any env var not set falls back to the dataclass default.

        Environment variables recognised
        --------------------------------
        SATELLITE_CACHE_DIR
        SATELLITE_PROCESSING_VERSION
        SATELLITE_SPATIAL_RESOLUTION_M
        SATELLITE_CLOUD_THRESHOLD
        """
        kwargs: dict = {}

        if res := os.environ.get("SATELLITE_SPATIAL_RESOLUTION_M"):
            kwargs["spatial_resolution_m"] = int(res)
        if ct := os.environ.get("SATELLITE_CLOUD_THRESHOLD"):
            kwargs["cloud_threshold"] = float(ct)

        return cls(**kwargs)
