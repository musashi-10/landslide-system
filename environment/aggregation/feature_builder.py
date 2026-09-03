"""
environment/aggregation/feature_builder.py

Main feature builder — the primary interface for Engineer 5 and the pipeline.

Public interface
----------------
build_dynamic_features(...)  — build a DynamicRecord for one location and time.
get_dynamic_conditions(...)  — convenience wrapper; same as above.

These are the functions that Engineer 5 (ML) calls.
Engineers 5 and 6 should NOT need to know about providers, ingestors,
or window calculation internals.

How it works
------------
1. Fetch the observation history covering the largest window (7d) using the provider.
2. Validate and ingest via RainfallIngestor.
3. Build a UTC-indexed Series.
4. Compute all rainfall windows using the feature functions.
5. Fetch and align forecast.
6. Compute moisture indicator.
7. Return a validated DynamicRecord.

Error handling
--------------
- Provider failures: caught; affected fields set to None.
- Insufficient history: fields that require more data than available are None.
- Invalid input coordinates: raises ValueError immediately.

Caching
-------
Computed DynamicRecords are cached by (location_id, timestamp_utc).
Observation DataFrames are cached by (lat, lon, start, end, provider).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd

from environment.caching.local_cache import LocalCache
from environment.config import DEFAULT_CONFIG, EnvironmentConfig
from environment.features.forecast_features import (
    align_forecast_to_horizon,
    calculate_forecast_rainfall,
)
from environment.features.moisture_indicator import estimate_moisture_indicator
from environment.features.rainfall_windows import (
    build_datetime_series,
    calculate_antecedent_rainfall,
    calculate_rainfall_1h,
    calculate_rainfall_24h,
    calculate_rainfall_3d,
    calculate_rainfall_6h,
    calculate_rainfall_7d,
    calculate_rainfall_anomaly,
    calculate_rainfall_intensity,
)
from environment.ingestion.rainfall_ingestor import RainfallIngestor
from environment.providers.base import ProviderError, WeatherProvider
from environment.schemas.dynamic_record import DynamicRecord, utc_now_iso

logger = logging.getLogger(__name__)

# Module-level default cache (shared across calls unless overridden)
_DEFAULT_CACHE = LocalCache(ttl_seconds=3600)


def build_dynamic_features(
    location_id: str,
    latitude: float,
    longitude: float,
    timestamp_utc: str,
    provider: WeatherProvider,
    config: Optional[EnvironmentConfig] = None,
    cache: Optional[LocalCache] = None,
) -> DynamicRecord:
    """
    Build a DynamicRecord of environmental trigger features for a location and time.

    This is the primary interface for Engineer 5 (ML model) and the pipeline.

    Parameters
    ----------
    location_id : str
        Project location identifier (e.g. "LOC_+27.1230_+088.4560").
    latitude : float
        WGS-84 latitude in decimal degrees.
    longitude : float
        WGS-84 longitude in decimal degrees.
    timestamp_utc : str
        ISO 8601 UTC string for the "current" time (e.g. "2026-09-02T12:00:00Z").
        Rainfall windows are computed looking BACK from this time.
    provider : WeatherProvider
        The data provider to use.
    config : EnvironmentConfig | None
        Pipeline configuration.  Uses DEFAULT_CONFIG if None.
    cache : LocalCache | None
        Cache instance.  Uses module-level default cache if None.

    Returns
    -------
    DynamicRecord
        All available dynamic features at this location and time.
        Fields for which data is insufficient or unavailable are None.

    Raises
    ------
    ValueError
        On invalid input coordinates.

    Examples
    --------
    >>> from environment.providers import MockWeatherProvider
    >>> provider = MockWeatherProvider()
    >>> record = build_dynamic_features(
    ...     location_id="LOC_+27.1230_+088.4560",
    ...     latitude=27.123,
    ...     longitude=88.456,
    ...     timestamp_utc="2026-09-02T12:00:00Z",
    ...     provider=provider,
    ... )
    >>> record.rainfall_24h is not None
    True
    """
    if not (-90 <= latitude <= 90):
        raise ValueError(f"latitude {latitude} out of range [-90, 90]")
    if not (-180 <= longitude <= 180):
        raise ValueError(f"longitude {longitude} out of range [-180, 180]")

    cfg = config or DEFAULT_CONFIG
    cache = cache or _DEFAULT_CACHE

    # Check feature cache
    feature_key = LocalCache.make_feature_key(location_id, timestamp_utc)
    cached = cache.get(feature_key)
    if cached is not None:
        logger.debug("Feature cache HIT for %s @ %s", location_id, timestamp_utc)
        return cached

    ref_dt = _parse_utc(timestamp_utc)
    acquisition_time = utc_now_iso()

    # --- 1. Fetch observation history (7-day window) ---
    max_window = max(cfg.window_hours.values())  # hours
    start_dt = ref_dt - timedelta(hours=max_window)
    start_utc = start_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    obs_key = LocalCache.make_observation_key(
        latitude, longitude, start_utc, timestamp_utc, provider.name
    )
    obs_df: Optional[pd.DataFrame] = cache.get(obs_key)

    if obs_df is None:
        ingestor = RainfallIngestor(provider, max_valid_hourly_mm=cfg.max_valid_hourly_rainfall_mm)
        obs_df = ingestor.ingest(latitude, longitude, start_utc, timestamp_utc)
        cache.set(obs_key, obs_df)

    # --- 2. Build time-indexed series ---
    series = _build_series(obs_df)

    ref_ts = pd.Timestamp(ref_dt)

    # --- 3. Compute rainfall windows ---
    rainfall_1h = _safe(calculate_rainfall_1h, series, ref_ts)
    rainfall_6h = _safe(calculate_rainfall_6h, series, ref_ts)
    rainfall_24h = _safe(calculate_rainfall_24h, series, ref_ts)
    rainfall_3d = _safe(calculate_rainfall_3d, series, ref_ts)
    rainfall_7d = _safe(calculate_rainfall_7d, series, ref_ts)
    rainfall_intensity = _safe_intensity(series, ref_ts)
    antecedent = _safe(calculate_antecedent_rainfall, series, ref_ts, cfg.moisture_decay_factor)
    rainfall_anomaly = _safe_anomaly(series, ref_ts)

    # --- 4. Moisture indicator ---
    moisture_indicator = estimate_moisture_indicator(antecedent)

    # --- 5. Fetch forecast ---
    fcast_key = LocalCache.make_forecast_key(latitude, longitude, cfg.forecast_horizon_hours, provider.name)
    forecast_records = cache.get(fcast_key)

    if forecast_records is None:
        try:
            forecast_records = provider.get_forecast(latitude, longitude, cfg.forecast_horizon_hours)
            cache.set(fcast_key, forecast_records)
        except ProviderError as exc:
            logger.warning(
                "Forecast unavailable for %s (%.4f, %.4f): %s — forecast_rainfall = None",
                location_id, latitude, longitude, exc,
            )
            forecast_records = []

    aligned_forecast = align_forecast_to_horizon(
        forecast_records, timestamp_utc, cfg.forecast_horizon_hours
    )
    forecast_rainfall = calculate_forecast_rainfall(aligned_forecast, cfg.forecast_horizon_hours)

    # --- 6. Build DynamicRecord ---
    record = DynamicRecord(
        location_id=location_id,
        latitude=latitude,
        longitude=longitude,
        timestamp_utc=timestamp_utc,
        rainfall_1h=rainfall_1h,
        rainfall_6h=rainfall_6h,
        rainfall_24h=rainfall_24h,
        rainfall_3d=rainfall_3d,
        rainfall_7d=rainfall_7d,
        forecast_rainfall=forecast_rainfall,
        rainfall_intensity=rainfall_intensity,
        antecedent_rainfall=antecedent,
        rainfall_anomaly=rainfall_anomaly,
        moisture_indicator=moisture_indicator,
        data_source=provider.name,
        acquisition_time_utc=acquisition_time,
        preprocessing_version=cfg.preprocessing_version,
    )

    cache.set(feature_key, record)
    logger.info(
        "Built DynamicRecord for %s @ %s | 24h=%.1f mm | 7d=%.1f mm | moisture=%.3f",
        location_id, timestamp_utc,
        rainfall_24h or 0.0,
        rainfall_7d or 0.0,
        moisture_indicator or 0.0,
    )
    return record


# Convenience alias (same signature as build_dynamic_features)
get_dynamic_conditions = build_dynamic_features


def dynamic_record_to_dict(record: DynamicRecord) -> dict:
    """
    Convert a DynamicRecord to a flat dictionary for JSON serialisation.

    This is the format expected by Engineer 6's API response.
    Only includes the data-contract fields (not provenance metadata).
    """
    return {
        "location_id": record.location_id,
        "latitude": record.latitude,
        "longitude": record.longitude,
        "timestamp_utc": record.timestamp_utc,
        "rainfall_1h": record.rainfall_1h,
        "rainfall_6h": record.rainfall_6h,
        "rainfall_24h": record.rainfall_24h,
        "rainfall_3d": record.rainfall_3d,
        "rainfall_7d": record.rainfall_7d,
        "forecast_rainfall": record.forecast_rainfall,
        "moisture_indicator": record.moisture_indicator,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_utc(ts: str) -> datetime:
    ts = ts.strip()
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _build_series(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)
    return build_datetime_series(df, "timestamp_utc", "rainfall_mm")


def _safe(fn, series: pd.Series, ref_ts: pd.Timestamp, *args) -> Optional[float]:
    """Call a window function; return None on any error."""
    if series.empty:
        return None
    try:
        return fn(series, *args, reference_time=ref_ts)
    except Exception as exc:
        logger.warning("Window function %s failed: %s", fn.__name__, exc)
        return None


def _safe_intensity(series: pd.Series, ref_ts: pd.Timestamp) -> Optional[float]:
    if series.empty:
        return None
    try:
        return calculate_rainfall_intensity(series, window_minutes=60, reference_time=ref_ts)
    except Exception as exc:
        logger.warning("Intensity calculation failed: %s", exc)
        return None


def _safe_anomaly(series: pd.Series, ref_ts: pd.Timestamp) -> Optional[float]:
    if series.empty:
        return None
    try:
        return calculate_rainfall_anomaly(series, window_hours=24, baseline_hours=168, reference_time=ref_ts)
    except Exception as exc:
        logger.debug("Anomaly calculation failed (insufficient history): %s", exc)
        return None
