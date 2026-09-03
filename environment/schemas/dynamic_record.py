"""
environment/schemas/dynamic_record.py

Pydantic models for the dynamic environmental-condition pipeline.

These schemas define the canonical data structures used within the
environment module.  The public handoff to Engineer 5 / Engineer 6 uses
``shared.schemas.EnvironmentalFeatures`` — see ``aggregation.feature_builder``
for the conversion helper.

Units
-----
All rainfall values are in **millimetres (mm)**.
``moisture_indicator`` is a dimensionless proxy in the range [0, 1].
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Raw observation (provider output, before aggregation)
# ---------------------------------------------------------------------------


class RawObservation(BaseModel):
    """A single hourly rainfall observation from a weather provider."""

    location_id: str
    latitude: float
    longitude: float
    timestamp_utc: str = Field(
        description="ISO 8601 UTC timestamp, e.g. 2026-09-02T12:00:00Z"
    )
    rainfall_mm: Optional[float] = Field(
        default=None,
        description=(
            "Hourly accumulated precipitation in millimetres (mm). "
            "None means observation is genuinely missing — never coerced to 0."
        ),
    )
    source: str = ""

    @field_validator("rainfall_mm")
    @classmethod
    def rainfall_must_be_non_negative(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and v < 0:
            raise ValueError(
                f"rainfall_mm must be >= 0 (physically impossible negative rainfall). Got {v}"
            )
        return v


# ---------------------------------------------------------------------------
# Forecast record (single future hour from a forecast provider)
# ---------------------------------------------------------------------------


class ForecastRecord(BaseModel):
    """A single forecast hour from a weather provider."""

    location_id: str
    latitude: float
    longitude: float
    timestamp_utc: str
    forecast_rainfall_mm: Optional[float] = Field(
        default=None,
        description=(
            "Forecast accumulated precipitation in millimetres (mm) for the hour. "
            "None means forecast is unavailable for this time step."
        ),
    )
    source: str = ""
    issued_at_utc: Optional[str] = None  # when the forecast was generated


# ---------------------------------------------------------------------------
# Canonical dynamic output record
# ---------------------------------------------------------------------------


class DynamicRecord(BaseModel):
    """
    Canonical dynamic feature record — the primary output of this pipeline.

    All downstream consumers (Engineer 5, Engineer 6) should work with this
    structure or with the ``shared.schemas.EnvironmentalFeatures`` merged record.

    Rainfall fields
    ---------------
    - rainfall_1h        : 1-hour accumulated precipitation (mm)
    - rainfall_6h        : 6-hour rolling accumulated precipitation (mm)
    - rainfall_24h       : 24-hour rolling accumulated precipitation (mm)
    - rainfall_3d        : 72-hour rolling accumulated precipitation (mm)
    - rainfall_7d        : 168-hour rolling accumulated precipitation (mm)
    - forecast_rainfall  : forecast accumulation over the configured horizon (mm)
    - rainfall_intensity : current hourly intensity (mm/hour) — rate, not accumulation

    Derived features
    ----------------
    - antecedent_rainfall : exponentially-weighted antecedent rainfall index
    - rainfall_anomaly    : deviation from recent mean (mm) — optional

    Moisture proxy
    --------------
    - moisture_indicator : dimensionless [0, 1] proxy for environmental wetness.
      This is NOT a physical soil-moisture measurement.  It is an estimated
      indicator derived from recent rainfall accumulation.  See
      ``environment/features/moisture_indicator.py`` for details and limitations.

    Missing values
    --------------
    ``None`` means the value could not be computed (insufficient history,
    provider failure, etc.).  Per the data contract, missing values must
    **never** silently become 0.
    """

    # ----- Spatial / temporal identity -----
    location_id: str
    latitude: float
    longitude: float
    timestamp_utc: str = Field(
        description="ISO 8601 UTC timestamp at which these features are valid"
    )

    # ----- Mandatory rainfall windows (data-contract §6) -----
    rainfall_1h: Optional[float] = Field(default=None, description="1-hour accumulated rainfall (mm)")
    rainfall_6h: Optional[float] = Field(default=None, description="6-hour rolling rainfall (mm)")
    rainfall_24h: Optional[float] = Field(default=None, description="24-hour rolling rainfall (mm)")
    rainfall_3d: Optional[float] = Field(default=None, description="3-day (72-hour) rolling rainfall (mm)")
    rainfall_7d: Optional[float] = Field(default=None, description="7-day (168-hour) rolling rainfall (mm)")
    forecast_rainfall: Optional[float] = Field(
        default=None,
        description="Forecast accumulated rainfall over configured horizon (mm). None if unavailable."
    )

    # ----- Optional additional features -----
    rainfall_intensity: Optional[float] = Field(
        default=None,
        description="Instantaneous intensity: rainfall rate over last hour (mm/hour)"
    )
    antecedent_rainfall: Optional[float] = Field(
        default=None,
        description="Exponentially-weighted antecedent rainfall index (dimensionless mm-equivalent)"
    )
    rainfall_anomaly: Optional[float] = Field(
        default=None,
        description="Departure of rainfall_24h from recent rolling mean (mm). Can be negative."
    )

    # ----- Moisture indicator (data-contract §6) -----
    moisture_indicator: Optional[float] = Field(
        default=None,
        description=(
            "Estimated environmental moisture proxy [0, 1]. "
            "Derived from recent rainfall history. "
            "NOT a physical soil-moisture measurement."
        ),
    )

    # ----- Provenance -----
    data_source: str = Field(default="", description="Provider name, e.g. 'open_meteo', 'mock'")
    acquisition_time_utc: Optional[str] = Field(
        default=None, description="UTC time this record was fetched/computed"
    )
    preprocessing_version: str = "1.0.0"


# ---------------------------------------------------------------------------
# Data provenance (attached to pipeline output batches)
# ---------------------------------------------------------------------------


class DataProvenance(BaseModel):
    """
    Provenance metadata for a batch of dynamic records.

    Every dynamic dataset must retain this information per the data contract §11.
    """

    source: str = Field(description="Provider / data source name")
    source_url: str = ""
    acquisition_time_utc: str = Field(description="UTC time data was fetched")
    geographic_coverage: str = Field(default="", description="Description of spatial extent")
    spatial_resolution: str = Field(
        description="E.g. '~1 km (ERA5-Land)', 'point observation', 'mock'"
    )
    temporal_resolution: str = Field(
        description="E.g. '1-hour accumulated', 'hourly observation'"
    )
    update_frequency: str = Field(
        description=(
            "How often this source updates, "
            "e.g. 'hourly (forecast)', '~5-day lag (reanalysis)'"
        )
    )
    units: str = "mm (millimetres)"
    rainfall_value_type: str = Field(
        description=(
            "Whether values represent 'accumulated_per_hour', 'rate_mm_per_hour', "
            "or 'forecast_accumulation'. Must never mix silently."
        )
    )
    preprocessing_version: str = "1.0.0"
    limitations: str = Field(default="", description="Known limitations of this data source")
    license: str = Field(default="", description="Usage license / restrictions")


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
