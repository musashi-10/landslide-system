"""
Internal data schemas for the data engineering pipeline.

These schemas describe the standardized intermediate and output records
produced by the ingestion/preprocessing pipeline.  They are distinct from
the shared/schemas layer so that internal representation can evolve without
breaking the inter-engineer contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class LandslideRecord:
    """
    One standardised historical landslide event record.

    Attributes
    ----------
    location_id : str
        Unique identifier for the spatial location (shared contract key).
    latitude : float
        WGS-84 latitude in decimal degrees.
    longitude : float
        WGS-84 longitude in decimal degrees.
    geometry : str | None
        WKT representation of the point geometry, or None when not available.
    timestamp_utc : str
        ISO 8601 timestamp in UTC (e.g. ``2026-09-02T12:00:00Z``).
    historical_landslide : int
        Binary label: ``1`` = landslide event recorded, ``0`` = non-event.
    source : str
        Human-readable data-source identifier (filename, database name, etc.).
    grid_id : str | None
        Grid cell identifier assigned during spatial indexing (optional).
    """

    location_id: str
    latitude: float
    longitude: float
    geometry: Optional[str]
    timestamp_utc: str
    historical_landslide: int
    source: str
    grid_id: Optional[str] = None


@dataclass
class ValidationReport:
    """
    Structured summary of a validation run over a dataset.

    Attributes
    ----------
    records_total : int
        Total records inspected.
    records_valid : int
        Records that passed all validation rules.
    records_invalid : int
        Records that failed at least one validation rule.
    missing_coordinates : int
        Records with a null/missing coordinate field.
    invalid_coordinates : int
        Records with coordinates outside the valid WGS-84 range.
    missing_timestamps : int
        Records where ``timestamp_utc`` could not be parsed or was absent.
    duplicate_records : int
        Records detected as likely duplicates of another record.
    missing_feature_values : int
        Records with at least one required feature column set to None.
    geometry_errors : int
        Records with invalid or un-parseable geometry.
    invalid_crs : int
        Geospatial files/records whose CRS could not be identified or
        converted.
    details : list[dict]
        Per-record detail items for invalid records.  Each entry is a
        ``{"row": int, "reason": str}`` dict.
    """

    records_total: int = 0
    records_valid: int = 0
    records_invalid: int = 0
    missing_coordinates: int = 0
    invalid_coordinates: int = 0
    missing_timestamps: int = 0
    duplicate_records: int = 0
    missing_feature_values: int = 0
    geometry_errors: int = 0
    invalid_crs: int = 0
    details: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Return a plain-dict representation (JSON-serialisable)."""
        return {
            "records_total": self.records_total,
            "records_valid": self.records_valid,
            "records_invalid": self.records_invalid,
            "missing_coordinates": self.missing_coordinates,
            "invalid_coordinates": self.invalid_coordinates,
            "missing_timestamps": self.missing_timestamps,
            "duplicate_records": self.duplicate_records,
            "missing_feature_values": self.missing_feature_values,
            "geometry_errors": self.geometry_errors,
            "invalid_crs": self.invalid_crs,
        }


@dataclass
class DataProvenance:
    """
    Provenance metadata for a processed dataset.

    Every externally-sourced dataset that passes through the pipeline must
    be accompanied by a ``DataProvenance`` record, as required by the data
    contract (section 11).

    Attributes
    ----------
    source : str
        Human-readable source name or URL.
    acquisition_datetime : str | None
        ISO 8601 UTC datetime when the data was acquired, or None.
    geographic_coverage : str | None
        Free-text or bounding-box description of the geographic area.
    spatial_resolution : str | None
        E.g. ``"30m"``, ``"0.01°"``.
    temporal_resolution : str | None
        E.g. ``"daily"``, ``"annual"``.
    preprocessing_version : str
        Version string of the preprocessing code that produced the output.
    license_notes : str | None
        Usage restrictions or licence notes, if any.
    """

    source: str
    acquisition_datetime: Optional[str] = None
    geographic_coverage: Optional[str] = None
    spatial_resolution: Optional[str] = None
    temporal_resolution: Optional[str] = None
    preprocessing_version: str = "1.0.0"
    license_notes: Optional[str] = None

    def to_dict(self) -> dict:
        """Return a plain-dict representation (JSON-serialisable)."""
        return {
            "source": self.source,
            "acquisition_datetime": self.acquisition_datetime,
            "geographic_coverage": self.geographic_coverage,
            "spatial_resolution": self.spatial_resolution,
            "temporal_resolution": self.temporal_resolution,
            "preprocessing_version": self.preprocessing_version,
            "license_notes": self.license_notes,
        }
