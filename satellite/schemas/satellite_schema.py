"""
satellite/schemas/satellite_schema.py
---------------------------------------
Pydantic model for the satellite-derived feature output record.

This is the OUTPUT CONTRACT of the satellite pipeline.
Every row in the Parquet feature table corresponds to one instance
of ``SatelliteFeatureRecord``.

Join key
--------
``location_id`` is the PRIMARY key for joining with:
- Engineer 2's GIS static features
- Engineer 5's ML feature table

Schema versioning
-----------------
``processing_version`` carries the pipeline version so downstream
consumers can handle schema evolution.

Missing values
--------------
All satellite-derived feature fields are Optional.  A missing value
must be ``None`` (Python / JSON null) — never 0.0 (per data contract
Section 12).
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, field_validator, model_validator


class SatelliteFeatureRecord(BaseModel):
    """One satellite-derived feature observation for one spatial location.

    Fields
    ------
    location_id : str
        Primary spatial join key.  Format: ``LOC_LAT{lat:.4f}_LON{lon:.4f}``.
    latitude : float
        Pixel centroid latitude in decimal degrees (EPSG:4326).
    longitude : float
        Pixel centroid longitude in decimal degrees (EPSG:4326).
    geometry : str, optional
        WKT representation of the point geometry, e.g. ``POINT (lon lat)``.
    acquisition_time : str
        ISO 8601 UTC timestamp of the satellite image acquisition.
        Example: ``2024-06-15T10:23:00Z``
    ndvi : float, optional
        Normalized Difference Vegetation Index.  Range [−1, 1].
    bare_surface_index : float, optional
        Bare Soil Index.  Range [−1, 1].  Higher = more bare/exposed ground.
    ndvi_change : float, optional
        NDVI delta between two acquisitions (t2 − t1).  Range (−2, 2).
        Negative = vegetation loss.  Only present for change-detection runs.
    land_cover : str, optional
        Rule-based land-cover category.  One of:
        ``forest``, ``shrub_grass``, ``bare``, ``water``, ``unknown``.
    source : str
        Satellite data source.  e.g. ``"Sentinel-2"``.
    spatial_resolution_m : int
        Output pixel resolution in metres.
    processing_version : str
        Version of the satellite feature pipeline used to produce this record.
    source_crs : str, optional
        CRS of the source raster before reprojection.
    output_crs : str
        CRS of the output coordinates.  Always ``"EPSG:4326"``.
    """

    # Spatial identity
    location_id: str
    latitude: float
    longitude: float
    geometry: Optional[str] = None

    # Temporal
    acquisition_time: str

    # Satellite-derived features (all optional — missing = None, not 0.0)
    ndvi: Optional[float] = None
    bare_surface_index: Optional[float] = None
    ndvi_change: Optional[float] = None
    land_cover: Optional[str] = None

    # Provenance (data-contract Section 5 & 11)
    source: str
    spatial_resolution_m: int
    processing_version: str
    source_crs: Optional[str] = None
    output_crs: str = "EPSG:4326"

    @field_validator("ndvi", "bare_surface_index")
    @classmethod
    def check_index_range(cls, v: Optional[float]) -> Optional[float]:
        """Validate that spectral indices are within [−1, 1]."""
        if v is not None and not (-1.0 <= v <= 1.0):
            raise ValueError(f"Spectral index out of range [−1, 1]: {v}")
        return v

    @field_validator("land_cover")
    @classmethod
    def check_land_cover(cls, v: Optional[str]) -> Optional[str]:
        """Validate that land cover is a known category."""
        valid = {"forest", "shrub_grass", "bare", "water", "unknown", None}
        if v not in valid:
            raise ValueError(
                f"Unknown land_cover value '{v}'.  Valid: {valid - {None}}"
            )
        return v

    @field_validator("output_crs")
    @classmethod
    def check_output_crs(cls, v: str) -> str:
        """Validate output CRS is EPSG:4326."""
        if v != "EPSG:4326":
            raise ValueError(
                f"output_crs must be 'EPSG:4326', got '{v}'.  "
                "All satellite outputs must use WGS-84."
            )
        return v

    @field_validator("spatial_resolution_m")
    @classmethod
    def check_resolution(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"spatial_resolution_m must be positive, got {v}")
        return v

    @model_validator(mode="after")
    def check_lat_lon(self) -> "SatelliteFeatureRecord":
        if not (-90.0 <= self.latitude <= 90.0):
            raise ValueError(f"latitude out of range: {self.latitude}")
        if not (-180.0 <= self.longitude <= 180.0):
            raise ValueError(f"longitude out of range: {self.longitude}")
        return self
