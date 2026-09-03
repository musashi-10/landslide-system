"""
environment/features/rainfall_windows.py

Rainfall accumulation window calculations.

All functions here are PURE — they depend only on their inputs and are
fully testable in isolation without a provider or database.

Input format
------------
Each function accepts a ``pd.Series`` with:
  - Index: timezone-aware pandas DatetimeTzDtype (UTC)
  - Values: float — hourly accumulated rainfall in mm.
             NaN represents missing/unavailable observations.

The series must be sorted in ascending time order before calling these functions.

Output
------
All functions return a ``float | None``.
``None`` is returned when there is insufficient data to compute the value
(e.g. the window has no non-NaN observations).
``None`` is NEVER silently converted to 0.

Rainfall value types
--------------------
- ``calculate_rolling_rainfall``   → accumulated sum (mm) — type: sum_accumulation
- ``calculate_rainfall_intensity`` → rate (mm/hour) — type: rate_mm_per_hour
- ``calculate_antecedent_rainfall``→ weighted index (mm-equivalent) — type: index

Units: millimetres (mm) unless explicitly noted.

Window sizes
------------
Window sizes are passed in as parameters, not hard-coded.
The canonical sizes come from ``environment.config.EnvironmentConfig.window_hours``.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# General rolling sum
# ---------------------------------------------------------------------------


def calculate_rolling_rainfall(
    series: pd.Series,
    window_hours: int,
    reference_time: Optional[pd.Timestamp] = None,
) -> Optional[float]:
    """
    Compute accumulated rainfall over the most recent ``window_hours`` hours.

    Parameters
    ----------
    series : pd.Series
        Hourly rainfall observations (mm), DatetimeTzDtype UTC index, sorted ascending.
    window_hours : int
        How many hours back to include (e.g. 24 for the last 24 hours).
    reference_time : pd.Timestamp | None
        The "current" time.  If None, uses the last timestamp in the series.

    Returns
    -------
    float | None
        Sum of non-NaN values in the window.  None if the window contains no
        valid observations (all NaN or window is empty).

    Notes
    -----
    Missing hours (NaN) are excluded from the sum but not imputed.
    A window with some NaN values will still return a sum — but the caller
    should be aware the sum represents only observed hours.
    If ALL values in the window are NaN, returns None.
    """
    if series.empty:
        return None

    series = _ensure_utc_sorted(series)
    ref = reference_time if reference_time is not None else series.index[-1]
    window_start = ref - pd.Timedelta(hours=window_hours)
    window = series[(series.index > window_start) & (series.index <= ref)]

    if window.empty or window.isna().all():
        return None

    total = float(window.sum(skipna=True))
    # Clamp to zero (floating point rounding can produce tiny negatives)
    return max(0.0, round(total, 3))


# ---------------------------------------------------------------------------
# Convenience wrappers for standard landslide trigger windows
# ---------------------------------------------------------------------------


def calculate_rainfall_1h(
    series: pd.Series,
    reference_time: Optional[pd.Timestamp] = None,
) -> Optional[float]:
    """1-hour accumulated rainfall (mm). Data contract: rainfall_1h."""
    return calculate_rolling_rainfall(series, window_hours=1, reference_time=reference_time)


def calculate_rainfall_6h(
    series: pd.Series,
    reference_time: Optional[pd.Timestamp] = None,
) -> Optional[float]:
    """6-hour accumulated rainfall (mm). Data contract: rainfall_6h."""
    return calculate_rolling_rainfall(series, window_hours=6, reference_time=reference_time)


def calculate_rainfall_24h(
    series: pd.Series,
    reference_time: Optional[pd.Timestamp] = None,
) -> Optional[float]:
    """24-hour accumulated rainfall (mm). Data contract: rainfall_24h."""
    return calculate_rolling_rainfall(series, window_hours=24, reference_time=reference_time)


def calculate_rainfall_3d(
    series: pd.Series,
    reference_time: Optional[pd.Timestamp] = None,
) -> Optional[float]:
    """3-day (72-hour) accumulated rainfall (mm). Data contract: rainfall_3d."""
    return calculate_rolling_rainfall(series, window_hours=72, reference_time=reference_time)


def calculate_rainfall_7d(
    series: pd.Series,
    reference_time: Optional[pd.Timestamp] = None,
) -> Optional[float]:
    """7-day (168-hour) accumulated rainfall (mm). Data contract: rainfall_7d."""
    return calculate_rolling_rainfall(series, window_hours=168, reference_time=reference_time)


# ---------------------------------------------------------------------------
# Rainfall intensity (rate)
# ---------------------------------------------------------------------------


def calculate_rainfall_intensity(
    series: pd.Series,
    window_minutes: int = 60,
    reference_time: Optional[pd.Timestamp] = None,
) -> Optional[float]:
    """
    Compute instantaneous rainfall intensity (mm/hour).

    This is the mean hourly rate over the specified ``window_minutes``.

    Parameters
    ----------
    series : pd.Series
        Hourly rainfall observations (mm).
    window_minutes : int
        Integration window in minutes.  Default: 60 (= last 1 hour).
        Must be a multiple of 60 for hourly data.
    reference_time : pd.Timestamp | None
        Reference time.

    Returns
    -------
    float | None
        Rainfall rate in mm/hour.  None if window is empty or all NaN.

    Notes
    -----
    For hourly data, intensity == rainfall_1h (both are mm over 1 hour).
    The function signature accepts ``window_minutes`` to be explicit about
    units and to support potential sub-hourly data in future.
    """
    if series.empty:
        return None

    series = _ensure_utc_sorted(series)
    ref = reference_time if reference_time is not None else series.index[-1]
    window_hours = window_minutes / 60.0
    window_start = ref - pd.Timedelta(minutes=window_minutes)
    window = series[(series.index > window_start) & (series.index <= ref)]

    if window.empty or window.isna().all():
        return None

    total = float(window.sum(skipna=True))
    if window_hours <= 0:
        return None

    # Intensity = total accumulation / window_hours = mm/hr
    intensity = max(0.0, total / window_hours)
    return round(intensity, 3)


# ---------------------------------------------------------------------------
# Antecedent rainfall (exponentially-weighted index)
# ---------------------------------------------------------------------------


def calculate_antecedent_rainfall(
    series: pd.Series,
    decay_factor: float = 0.85,
    reference_time: Optional[pd.Timestamp] = None,
) -> Optional[float]:
    """
    Compute an exponentially-weighted antecedent rainfall index.

    Antecedent rainfall (also called the antecedent precipitation index, API)
    captures the "memory" of past rainfall — wetter antecedent conditions
    reduce the infiltration capacity and increase landslide susceptibility.

    The index is computed as::

        API_t = sum_{k=0}^{N} decay_factor^k * rainfall_{t-k}

    where k=0 is the most recent hour and N is the length of the series.

    Parameters
    ----------
    series : pd.Series
        Hourly rainfall observations (mm), sorted ascending (oldest first).
    decay_factor : float
        Decay factor per hour.  Range: (0, 1).
        Higher values give more weight to recent rainfall and less to older.
        Default: 0.85 (moderate memory; ~1-week effective window).
    reference_time : pd.Timestamp | None
        Reference time.

    Returns
    -------
    float | None
        Antecedent rainfall index (dimensionless mm-equivalent).
        None if the series contains no valid observations.

    Scientific note
    ---------------
    The choice of decay factor is configurable.  A decay factor of 0.85/hr
    is commonly used for daily aggregation in literature (e.g. Kohler & Linsley,
    1951), adapted here for hourly data.  For operational use, this parameter
    should be calibrated against local soil types and drainage characteristics.
    """
    if series.empty:
        return None

    series = _ensure_utc_sorted(series)
    ref = reference_time if reference_time is not None else series.index[-1]

    # Work with values up to reference_time, oldest first
    window = series[series.index <= ref].copy()
    if window.empty or window.isna().all():
        return None

    # Reverse so index 0 = most recent
    values = window.values[::-1].astype(float)

    api = 0.0
    for k, v in enumerate(values):
        if not np.isnan(v):
            api += (decay_factor ** k) * v

    return max(0.0, round(api, 3))


# ---------------------------------------------------------------------------
# Rainfall anomaly
# ---------------------------------------------------------------------------


def calculate_rainfall_anomaly(
    series: pd.Series,
    window_hours: int = 24,
    baseline_hours: int = 168,
    reference_time: Optional[pd.Timestamp] = None,
) -> Optional[float]:
    """
    Compute rainfall anomaly: departure of recent rainfall from a rolling baseline.

    Anomaly = rainfall_window - mean(hourly_rainfall_over_baseline_period)

    Parameters
    ----------
    series : pd.Series
        Hourly rainfall observations (mm).
    window_hours : int
        Recent accumulation window (default 24h).
    baseline_hours : int
        Baseline period for computing mean (default 168h = 7 days).
    reference_time : pd.Timestamp | None
        Reference time.

    Returns
    -------
    float | None
        Anomaly in mm.  Can be negative (drier than baseline).
        None if insufficient data.

    Notes
    -----
    This is a simple arithmetic anomaly, not a climatological anomaly.
    A climatological baseline would require multi-year historical data not
    available in the current prototype.
    """
    if series.empty:
        return None

    series = _ensure_utc_sorted(series)
    ref = reference_time if reference_time is not None else series.index[-1]

    recent = calculate_rolling_rainfall(series, window_hours=window_hours, reference_time=ref)
    if recent is None:
        return None

    baseline_start = ref - pd.Timedelta(hours=baseline_hours)
    baseline_window = series[(series.index > baseline_start) & (series.index <= ref)]
    valid_baseline = baseline_window.dropna()

    if valid_baseline.empty:
        return None

    # Mean hourly rate over baseline, scaled to recent window
    baseline_mean_rate = float(valid_baseline.mean())
    baseline_accumulation = baseline_mean_rate * window_hours
    anomaly = recent - baseline_accumulation
    return round(anomaly, 3)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ensure_utc_sorted(series: pd.Series) -> pd.Series:
    """Ensure the series has a UTC DatetimeTzDtype index and is sorted ascending."""
    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError(
            f"Series index must be DatetimeIndex, got {type(series.index).__name__}"
        )
    if series.index.tz is None:
        series = series.copy()
        series.index = series.index.tz_localize("UTC")
    else:
        series = series.copy()
        series.index = series.index.tz_convert("UTC")

    if not series.index.is_monotonic_increasing:
        series = series.sort_index()

    return series


def build_datetime_series(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp_utc",
    value_col: str = "rainfall_mm",
) -> pd.Series:
    """
    Build a UTC-indexed Series from a DataFrame for use with the window functions.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with timestamp and rainfall columns.
    timestamp_col : str
        Column containing ISO 8601 UTC strings.
    value_col : str
        Column containing rainfall values (float, NaN for missing).

    Returns
    -------
    pd.Series
        Float series with UTC DatetimeIndex, sorted ascending.
    """
    if df.empty:
        return pd.Series(dtype=float)

    index = pd.to_datetime(df[timestamp_col], utc=True)
    series = pd.Series(df[value_col].values, index=index, dtype=float)
    series.sort_index(inplace=True)
    return series
