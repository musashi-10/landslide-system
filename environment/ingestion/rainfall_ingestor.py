"""
environment/ingestion/rainfall_ingestor.py

Rainfall ingestion layer.

This module is responsible for:
1. Calling a WeatherProvider to fetch raw observations.
2. Validating each raw observation (non-negative, plausible range, valid timestamp).
3. Normalising timestamps to UTC ISO 8601.
4. Mapping raw coordinates to a project ``location_id`` using Engineer 1's format.
5. Returning a clean pandas DataFrame ready for feature computation.

Output DataFrame columns
-------------------------
location_id   : str   — project location identifier
latitude      : float — WGS-84 latitude (decimal degrees)
longitude     : float — WGS-84 longitude (decimal degrees)
timestamp_utc : str   — ISO 8601 UTC (e.g. "2026-09-02T12:00:00Z")
rainfall_mm   : float | NaN — hourly accumulated precipitation (mm)
                              NaN for genuinely missing observations
source        : str   — provider identifier

Note: ``NaN`` is used internally in the DataFrame for missing values.
The DynamicRecord output uses ``None`` (Pydantic Optional).
Neither is silently coerced to 0.

Spatial grid note
-----------------
Provider data arrives at the nearest grid point to the requested coordinates.
This module re-assigns ``location_id`` using the *requested* coordinates, not
the provider grid.  The spatial mapping from provider grid to project grid is
documented in ``environment/preprocessing/spatial_mapper.py``.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import pandas as pd

from environment.providers.base import ProviderError, WeatherProvider
from environment.schemas.dynamic_record import RawObservation

logger = logging.getLogger(__name__)

# Maximum plausible hourly precipitation (mm).
# Above this value, the observation is flagged (not discarded).
# Reference: highest reliably recorded hourly rainfall ~300 mm/h (extreme events).
# For gridded products, 200 mm/h is a generous upper bound.
_MAX_PLAUSIBLE_HOURLY_MM = 200.0


class RainfallIngestor:
    """
    Ingest hourly rainfall observations from a provider into a clean DataFrame.

    Parameters
    ----------
    provider : WeatherProvider
        The data provider to use.
    max_valid_hourly_mm : float
        Upper plausibility limit.  Values above this are logged as warnings
        but kept in the output (not discarded).
    """

    def __init__(
        self,
        provider: WeatherProvider,
        max_valid_hourly_mm: float = _MAX_PLAUSIBLE_HOURLY_MM,
    ) -> None:
        self._provider = provider
        self._max_valid_hourly_mm = max_valid_hourly_mm

    def ingest(
        self,
        latitude: float,
        longitude: float,
        start_utc: str,
        end_utc: str,
    ) -> pd.DataFrame:
        """
        Fetch and validate observations, returning a clean DataFrame.

        Parameters
        ----------
        latitude : float
            Requested location latitude (WGS-84, decimal degrees).
        longitude : float
            Requested location longitude (WGS-84, decimal degrees).
        start_utc : str
            Start of the observation window (ISO 8601 UTC).
        end_utc : str
            End of the observation window (ISO 8601 UTC).

        Returns
        -------
        pd.DataFrame
            Columns: location_id, latitude, longitude, timestamp_utc, rainfall_mm, source
            Indexed by integer (not timestamp — caller reindexes as needed).
            Sorted by timestamp_utc ascending.
            Missing values are NaN (not 0).

        Notes
        -----
        On provider failure this method logs the error and returns an empty
        DataFrame.  The pipeline must handle the empty-DataFrame case.
        """
        try:
            raw_obs = self._provider.get_observations(latitude, longitude, start_utc, end_utc)
        except ProviderError as exc:
            logger.error(
                "Provider '%s' failed for (%.4f, %.4f) [%s -> %s]: %s",
                self._provider.name, latitude, longitude, start_utc, end_utc, exc,
            )
            return self._empty_dataframe()

        if not raw_obs:
            logger.warning(
                "Provider '%s' returned 0 observations for (%.4f, %.4f) [%s -> %s]",
                self._provider.name, latitude, longitude, start_utc, end_utc,
            )
            return self._empty_dataframe()

        rows = []
        n_invalid_ts = 0
        n_suspect = 0

        for obs in raw_obs:
            # Timestamp validation
            try:
                _validate_timestamp(obs.timestamp_utc)
            except ValueError as exc:
                logger.warning("Skipping observation with invalid timestamp %r: %s", obs.timestamp_utc, exc)
                n_invalid_ts += 1
                continue

            # Plausibility check (do not discard — just log)
            if obs.rainfall_mm is not None and obs.rainfall_mm > self._max_valid_hourly_mm:
                logger.warning(
                    "Suspect rainfall_mm=%.1f at %s (%.4f, %.4f) — exceeds %.0f mm/h threshold",
                    obs.rainfall_mm, obs.timestamp_utc, latitude, longitude, self._max_valid_hourly_mm,
                )
                n_suspect += 1

            rows.append(
                {
                    "location_id": _make_loc_id(latitude, longitude),
                    "latitude": latitude,
                    "longitude": longitude,
                    "timestamp_utc": obs.timestamp_utc,
                    "rainfall_mm": obs.rainfall_mm,  # None -> NaN via pd.DataFrame
                    "source": obs.source or self._provider.name,
                }
            )

        if n_invalid_ts:
            logger.info("Skipped %d observations with invalid timestamps", n_invalid_ts)
        if n_suspect:
            logger.info("Flagged %d suspect high-rainfall observations (kept in output)", n_suspect)

        if not rows:
            logger.warning("All observations were invalid for (%.4f, %.4f)", latitude, longitude)
            return self._empty_dataframe()

        df = pd.DataFrame(rows)
        df["rainfall_mm"] = pd.to_numeric(df["rainfall_mm"], errors="coerce")  # None -> NaN
        df.sort_values("timestamp_utc", inplace=True)
        df.reset_index(drop=True, inplace=True)

        logger.debug(
            "Ingestor: %d valid rows for (%.4f, %.4f)", len(df), latitude, longitude
        )
        return df

    @staticmethod
    def _empty_dataframe() -> pd.DataFrame:
        return pd.DataFrame(
            columns=["location_id", "latitude", "longitude", "timestamp_utc", "rainfall_mm", "source"]
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_timestamp(ts: str) -> None:
    """Raise ValueError if ts is not a parseable ISO 8601 UTC string."""
    if not ts or not ts.strip():
        raise ValueError("Empty timestamp")
    ts_clean = ts.strip()
    if ts_clean.endswith("Z"):
        ts_clean = ts_clean[:-1] + "+00:00"
    try:
        datetime.fromisoformat(ts_clean)
    except ValueError as exc:
        raise ValueError(f"Cannot parse timestamp {ts!r}: {exc}") from exc


def _make_loc_id(lat: float, lon: float) -> str:
    """
    Generate location_id compatible with Engineer 1's format.

    Format: LOC_{lat:+.4f}_{lon:+08.4f}
    Example: LOC_+27.1230_+088.4560
    """
    return f"LOC_{lat:+.4f}_{lon:+08.4f}"
