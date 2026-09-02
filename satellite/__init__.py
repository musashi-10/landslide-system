"""
satellite — Satellite imagery and remote-sensing feature pipeline.

Public API
----------
The primary interface for other engineers:

    from satellite import extract_satellite_features, extract_features_from_arrays
    from satellite.schemas import SatelliteFeatureRecord
    from satellite.config import SatelliteConfig

For full usage instructions see satellite/SATELLITE_PIPELINE.md.
"""

from .pipeline import extract_satellite_features, extract_features_from_arrays
from .schemas.satellite_schema import SatelliteFeatureRecord
from .config import SatelliteConfig

__all__ = [
    "extract_satellite_features",
    "extract_features_from_arrays",
    "SatelliteFeatureRecord",
    "SatelliteConfig",
]
