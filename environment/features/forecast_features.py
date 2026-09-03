"""
environment/features/forecast_features.py

Forecast rainfall feature computation.

This module converts a list of ``ForecastRecord`` objects into a single
``forecast_rainfall`` value — the total accumulated forecast precipitation
over the configured horizon.

Missing data handling
---------------------
- If no forecast records are provided → returns None (not 0).
- If all records in the horizon have None precipitation → returns None (not 0).
- If some records have None and some have valid values → sums available values
  and logs a warning about incomplete coverage.

This conservative approach prevents the pipeline from silently under-reporting
expected rainfall (which could suppress landslide risk warnings).

Value type
----------
forecast_rainfall represents *forecast accumulation over horizon_hours* (mm).
This is NOT a rate.  Document this when handing off to Engineer 5.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from environment.schemas.dynamic_record import ForecastRecord

logger = logging.getLogger(__name__)


def calculate_forecast_rainfall(
    forecast_records: List[ForecastRecord],
    horizon_hours: int = 24,
) -> Optional[float]:
    """
    Compute total forecast accumulated rainfall over the next ``horizon_hours``.

    Parameters
    ----------
    forecast_records : list[ForecastRecord]
        Ordered list of future hourly forecast records.
        May be empty if the provider returned no forecast.
    horizon_hours : int
        How many future hours to include in the accumulation.
        Default: 24 hours.

    Returns
    -------
    float | None
        Total accumulated forecast rainfall in mm over ``horizon_hours``.
        Returns None if:
          - ``forecast_records`` is empty
          - All records within the horizon have None precipitation
        Returns a partial sum (with warning) if only some records are None.

    Notes
    -----
    Missing forecast hours are NOT imputed as zero.
    The caller (Engineer 5) should be aware that a partial sum underestimates
    true accumulation.
    """
    if not forecast_records:
        logger.debug("No forecast records provided; forecast_rainfall = None")
        return None

    # Take only the first horizon_hours records
    horizon_records = forecast_records[:horizon_hours]

    valid_values: List[float] = []
    missing_count = 0

    for rec in horizon_records:
        if rec.forecast_rainfall_mm is None:
            missing_count += 1
        else:
            valid_values.append(rec.forecast_rainfall_mm)

    if not valid_values:
        logger.debug(
            "All %d forecast records have None precipitation; forecast_rainfall = None",
            len(horizon_records),
        )
        return None

    if missing_count > 0:
        logger.warning(
            "%d of %d forecast hours are missing; forecast_rainfall is a partial sum "
            "(underestimates true accumulation)",
            missing_count, len(horizon_records),
        )

    total = sum(valid_values)
    return max(0.0, round(total, 3))


def align_forecast_to_horizon(
    forecast_records: List[ForecastRecord],
    reference_utc: str,
    horizon_hours: int,
) -> List[ForecastRecord]:
    """
    Filter forecast records to those within ``horizon_hours`` of ``reference_utc``.

    Parameters
    ----------
    forecast_records : list[ForecastRecord]
        Full forecast list (may span many hours).
    reference_utc : str
        The reference timestamp (ISO 8601 UTC) representing "now".
    horizon_hours : int
        How many hours ahead to include.

    Returns
    -------
    list[ForecastRecord]
        Filtered, time-ordered records within the horizon.
    """
    from datetime import datetime, timezone, timedelta

    def _parse(ts: str) -> datetime:
        ts = ts.strip()
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        dt = datetime.fromisoformat(ts)
        return dt.astimezone(timezone.utc)

    try:
        ref_dt = _parse(reference_utc)
    except (ValueError, AttributeError) as exc:
        logger.warning("Could not parse reference_utc %r: %s", reference_utc, exc)
        return []

    horizon_end = ref_dt + timedelta(hours=horizon_hours)
    aligned = [
        rec for rec in forecast_records
        if _parse(rec.timestamp_utc) > ref_dt and _parse(rec.timestamp_utc) <= horizon_end
    ]
    aligned.sort(key=lambda r: r.timestamp_utc)
    return aligned
