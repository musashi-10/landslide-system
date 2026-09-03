"""
environment/providers/open_meteo_provider.py

Open-Meteo weather provider.

Open-Meteo (https://open-meteo.com) is a free, open-source weather API that
provides historical reanalysis and real-time forecast data.  No API key is
required for standard use.

Data characteristics
--------------------
Source         : Open-Meteo API (ERA5-Land reanalysis + ICON forecast)
Spatial resol. : ~1 km (ERA5-Land) / ~2–13 km (ICON NWP forecast)
Temporal resol.: Hourly observations
Update freq.   : ERA5-Land updates with ~5-day lag; forecast updates every 6h
Units          : mm (accumulated per hour — ``precipitation`` variable)
License        : CC BY 4.0 (attribution required for commercial use)

Limitations
-----------
- ERA5-Land reanalysis has a ~5-day lag; very recent hours fall back to
  forecast fields which are less accurate.
- Grid spacing (~1 km) may miss sub-kilometre variability.
- The API has rate limits; this provider respects them with backoff.
- Open-Meteo data represents *grid-cell-average* precipitation, not a point
  raingauge reading.  Actual local accumulations may differ.
- ``precipitation`` is accumulated over each 1-hour period (mm/h as mm).

API docs: https://open-meteo.com/en/docs
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import requests

from environment.providers.base import (
    ProviderError,
    ProviderInvalidResponseError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    WeatherProvider,
)
from environment.schemas.dynamic_record import (
    DataProvenance,
    ForecastRecord,
    RawObservation,
    utc_now_iso,
)

logger = logging.getLogger(__name__)

_HISTORICAL_URL = "https://archive-api.open-meteo.com/v1/archive"
_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

_DEFAULT_TIMEOUT_S = 30
_MAX_RETRIES = 3
_BACKOFF_BASE_S = 2.0  # exponential backoff base


class OpenMeteoProvider(WeatherProvider):
    """
    Weather provider backed by the Open-Meteo free API.

    Parameters
    ----------
    timeout_seconds : int
        HTTP request timeout.
    max_retries : int
        Number of retries on transient failures.
    session : requests.Session | None
        Optional pre-configured session (useful for testing with mocked HTTP).
    """

    def __init__(
        self,
        timeout_seconds: int = _DEFAULT_TIMEOUT_S,
        max_retries: int = _MAX_RETRIES,
        session: Optional[requests.Session] = None,
    ) -> None:
        self._timeout = timeout_seconds
        self._max_retries = max_retries
        self._session = session or requests.Session()

    # ------------------------------------------------------------------
    # WeatherProvider interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "open_meteo"

    @property
    def provenance(self) -> DataProvenance:
        return DataProvenance(
            source="open_meteo",
            source_url="https://open-meteo.com",
            acquisition_time_utc=utc_now_iso(),
            geographic_coverage="Global (ERA5-Land / ICON NWP)",
            spatial_resolution="~1 km (ERA5-Land reanalysis); ~2–13 km (ICON forecast)",
            temporal_resolution="1-hour accumulated precipitation (mm/hr)",
            update_frequency=(
                "ERA5-Land: ~5-day lag from real-time; "
                "Forecast (ICON): updated every 6 hours"
            ),
            rainfall_value_type="accumulated_per_hour",
            limitations=(
                "ERA5-Land has a ~5-day reanalysis lag. "
                "Recent hours fall back to forecast data (lower accuracy). "
                "Grid-cell-average values may differ from point raingauge readings. "
                "Subject to Open-Meteo rate limits (free tier). "
                "Spatial resolution (~1 km) may miss sub-km rainfall variability."
            ),
            license="CC BY 4.0 (attribution required for commercial use)",
        )

    def get_observations(
        self,
        latitude: float,
        longitude: float,
        start_utc: str,
        end_utc: str,
    ) -> List[RawObservation]:
        """
        Fetch hourly precipitation from Open-Meteo archive API.

        Uses ERA5-Land reanalysis.  For dates within the last 5 days,
        the API falls back to model data (NWP) transparently.
        """
        start_dt = _parse_utc(start_utc)
        end_dt = _parse_utc(end_utc)

        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_dt.strftime("%Y-%m-%d"),
            "end_date": end_dt.strftime("%Y-%m-%d"),
            "hourly": "precipitation",
            "timezone": "UTC",
        }

        data = self._get_with_retry(_HISTORICAL_URL, params)

        try:
            times = data["hourly"]["time"]
            precip = data["hourly"]["precipitation"]
        except (KeyError, TypeError) as exc:
            raise ProviderInvalidResponseError(
                f"Open-Meteo archive response missing 'hourly.time' or 'hourly.precipitation': {exc}"
            ) from exc

        loc_id = _make_loc_id(latitude, longitude)
        result: List[RawObservation] = []

        for ts_str, val in zip(times, precip):
            try:
                ts_dt = _parse_naive_utc(ts_str)
            except ValueError as exc:
                logger.warning("Could not parse timestamp %r from Open-Meteo: %s", ts_str, exc)
                continue

            # Filter to the exact requested window
            if not (start_dt <= ts_dt <= end_dt):
                continue

            rain: Optional[float] = None
            if val is not None:
                try:
                    rain_f = float(val)
                    # Open-Meteo returns negative values for trace/rounding — clamp to 0
                    rain = max(0.0, rain_f)
                except (ValueError, TypeError):
                    pass  # rain stays None

            result.append(
                RawObservation(
                    location_id=loc_id,
                    latitude=latitude,
                    longitude=longitude,
                    timestamp_utc=ts_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    rainfall_mm=rain,
                    source="open_meteo_archive",
                )
            )

        result.sort(key=lambda r: r.timestamp_utc)
        logger.debug(
            "OpenMeteo archive: %d observations for (%.4f, %.4f) [%s -> %s]",
            len(result), latitude, longitude, start_utc, end_utc,
        )
        return result

    def get_forecast(
        self,
        latitude: float,
        longitude: float,
        horizon_hours: int,
    ) -> List[ForecastRecord]:
        """
        Fetch hourly precipitation forecast from Open-Meteo forecast API.

        Uses ICON NWP model.  Horizon is capped at the API maximum (16 days).
        """
        horizon_hours = min(horizon_hours, 16 * 24)  # API cap

        params = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": "precipitation",
            "timezone": "UTC",
            "forecast_days": max(1, math.ceil(horizon_hours / 24)),
        }

        try:
            data = self._get_with_retry(_FORECAST_URL, params)
        except ProviderError as exc:
            logger.warning(
                "OpenMeteo forecast unavailable for (%.4f, %.4f): %s — returning empty list",
                latitude, longitude, exc,
            )
            return []

        try:
            times = data["hourly"]["time"]
            precip = data["hourly"]["precipitation"]
        except (KeyError, TypeError) as exc:
            raise ProviderInvalidResponseError(
                f"Open-Meteo forecast response malformed: {exc}"
            ) from exc

        now_utc = datetime.now(timezone.utc)
        issued_at = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
        loc_id = _make_loc_id(latitude, longitude)
        result: List[ForecastRecord] = []

        for ts_str, val in zip(times, precip):
            try:
                ts_dt = _parse_naive_utc(ts_str)
            except ValueError:
                continue

            # Only keep future hours within the requested horizon
            delta = (ts_dt - now_utc).total_seconds() / 3600.0
            if delta < 0 or delta > horizon_hours:
                continue

            rain: Optional[float] = None
            if val is not None:
                try:
                    rain = max(0.0, float(val))
                except (ValueError, TypeError):
                    pass

            result.append(
                ForecastRecord(
                    location_id=loc_id,
                    latitude=latitude,
                    longitude=longitude,
                    timestamp_utc=ts_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    forecast_rainfall_mm=rain,
                    source="open_meteo_forecast",
                    issued_at_utc=issued_at,
                )
            )

        result.sort(key=lambda r: r.timestamp_utc)
        logger.debug(
            "OpenMeteo forecast: %d records for (%.4f, %.4f), horizon=%dh",
            len(result), latitude, longitude, horizon_hours,
        )
        return result

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _get_with_retry(self, url: str, params: dict) -> dict:
        """GET with exponential backoff for transient failures."""
        last_exc: Optional[Exception] = None

        for attempt in range(1, self._max_retries + 1):
            try:
                resp = self._session.get(url, params=params, timeout=self._timeout)
            except requests.exceptions.Timeout as exc:
                last_exc = exc
                logger.warning("OpenMeteo timeout (attempt %d/%d): %s", attempt, self._max_retries, exc)
                self._backoff(attempt)
                continue
            except requests.exceptions.ConnectionError as exc:
                last_exc = exc
                logger.warning("OpenMeteo connection error (attempt %d/%d): %s", attempt, self._max_retries, exc)
                self._backoff(attempt)
                continue

            if resp.status_code == 429:
                logger.warning("OpenMeteo rate limit (attempt %d/%d)", attempt, self._max_retries)
                self._backoff(attempt)
                last_exc = ProviderRateLimitError("Rate limit exceeded (HTTP 429)")
                continue

            if resp.status_code >= 500:
                logger.warning(
                    "OpenMeteo server error %d (attempt %d/%d)", resp.status_code, attempt, self._max_retries
                )
                self._backoff(attempt)
                last_exc = ProviderUnavailableError(f"HTTP {resp.status_code}")
                continue

            if resp.status_code >= 400:
                raise ProviderInvalidResponseError(
                    f"Open-Meteo client error {resp.status_code}: {resp.text[:200]}"
                )

            try:
                return resp.json()
            except ValueError as exc:
                raise ProviderInvalidResponseError(
                    f"Open-Meteo non-JSON response: {resp.text[:200]}"
                ) from exc

        raise ProviderTimeoutError(
            f"Open-Meteo failed after {self._max_retries} attempts"
        ) from last_exc

    @staticmethod
    def _backoff(attempt: int) -> None:
        sleep_s = _BACKOFF_BASE_S ** attempt
        logger.debug("Backoff %.1f seconds before retry", sleep_s)
        time.sleep(sleep_s)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

import math  # noqa: E402 (placed after class to keep class definition readable)


def _parse_utc(ts: str) -> datetime:
    ts = ts.strip()
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _parse_naive_utc(ts: str) -> datetime:
    """Parse Open-Meteo's naive UTC timestamps like '2026-09-01T00:00'."""
    dt = datetime.fromisoformat(ts.strip())
    return dt.replace(tzinfo=timezone.utc)


def _make_loc_id(lat: float, lon: float) -> str:
    return f"LOC_{lat:+.4f}_{lon:+08.4f}"
