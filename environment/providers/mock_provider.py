"""
environment/providers/mock_provider.py

Mock weather provider for testing and offline development.

The mock provider is deterministic — given the same location and time range
it always returns the same data.  It can operate in two modes:

1. **CSV mode** (default in tests):
   Reads from ``environment/tests/fixtures/synthetic_rainfall.csv``.
   This file contains an 8-day hourly time series for two locations.

2. **Generator mode** (fallback):
   Generates a sinusoidal + random rainfall pattern based on the location
   coordinates as a seed.  No external files required.

The mock provider never raises ``ProviderError`` unless explicitly configured
to simulate failures (via ``fail_mode``).

Usage
-----
>>> from environment.providers.mock_provider import MockWeatherProvider
>>> p = MockWeatherProvider()
>>> obs = p.get_observations(27.123, 88.456, "2026-09-01T00:00:00Z", "2026-09-01T05:00:00Z")
>>> len(obs) == 6
True
"""

from __future__ import annotations

import csv
import logging
import math
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

from environment.providers.base import (
    ProviderError,
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

# Default fixture path (relative to project root)
_DEFAULT_FIXTURE = (
    Path(__file__).parent.parent / "tests" / "fixtures" / "synthetic_rainfall.csv"
)


class MockWeatherProvider(WeatherProvider):
    """
    Deterministic mock provider for testing and offline development.

    Parameters
    ----------
    fixture_path : Path | None
        Path to a CSV file with columns:
        ``location_id, latitude, longitude, timestamp_utc, rainfall_mm``.
        If None, falls back to the bundled synthetic fixture.
    fail_mode : str | None
        Simulate failure modes for error-handling tests:
        - 'timeout'      -> raises ProviderTimeoutError
        - 'unavailable'  -> raises ProviderUnavailableError
        - None (default) -> normal operation
    """

    def __init__(
        self,
        fixture_path: Optional[Path] = None,
        fail_mode: Optional[str] = None,
    ) -> None:
        self._fixture_path = fixture_path or _DEFAULT_FIXTURE
        self._fail_mode = fail_mode
        self._fixture_data: Optional[list] = None

    # ------------------------------------------------------------------
    # WeatherProvider interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "mock"

    @property
    def provenance(self) -> DataProvenance:
        return DataProvenance(
            source="mock",
            source_url="n/a",
            acquisition_time_utc=utc_now_iso(),
            geographic_coverage="Synthetic test data",
            spatial_resolution="point (synthetic)",
            temporal_resolution="1-hour accumulated (synthetic)",
            update_frequency="on-demand (synthetic generation)",
            rainfall_value_type="accumulated_per_hour",
            limitations=(
                "Synthetic data for testing only.  "
                "Not based on real meteorological observations."
            ),
            license="internal test fixture",
        )

    def get_observations(
        self,
        latitude: float,
        longitude: float,
        start_utc: str,
        end_utc: str,
    ) -> List[RawObservation]:
        self._maybe_fail()

        start_dt = _parse_utc(start_utc)
        end_dt = _parse_utc(end_utc)

        # Try CSV fixture first
        if self._fixture_path.exists():
            return self._observations_from_fixture(latitude, longitude, start_dt, end_dt)

        # Fallback: generate deterministic sinusoidal data
        logger.warning(
            "Mock fixture not found at %s — using generated data", self._fixture_path
        )
        return self._generate_observations(latitude, longitude, start_dt, end_dt)

    def get_forecast(
        self,
        latitude: float,
        longitude: float,
        horizon_hours: int,
    ) -> List[ForecastRecord]:
        self._maybe_fail()

        now_utc = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        records: List[ForecastRecord] = []
        seed = int(abs(latitude * 1000 + longitude * 100)) % (2**31)
        rng = random.Random(seed + now_utc.hour)

        for h in range(horizon_hours):
            ts = now_utc + timedelta(hours=h + 1)
            ts_str = ts.strftime("%Y-%m-%dT%H:%M:%SZ")
            # Sinusoidal forecast with small random noise
            base = 5.0 + 4.0 * math.sin(h * math.pi / 12)
            val = max(0.0, base + rng.uniform(-2.0, 2.0))
            records.append(
                ForecastRecord(
                    location_id=_make_loc_id(latitude, longitude),
                    latitude=latitude,
                    longitude=longitude,
                    timestamp_utc=ts_str,
                    forecast_rainfall_mm=round(val, 2),
                    source="mock",
                    issued_at_utc=now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
                )
            )
        return records

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _maybe_fail(self) -> None:
        """Raise appropriate error if fail_mode is set."""
        from environment.providers.base import ProviderTimeoutError

        if self._fail_mode == "timeout":
            raise ProviderTimeoutError("Mock: simulated timeout")
        if self._fail_mode == "unavailable":
            raise ProviderUnavailableError("Mock: simulated service unavailable")

    def _load_fixture(self) -> list:
        """Load and cache CSV fixture data."""
        if self._fixture_data is not None:
            return self._fixture_data

        rows = []
        with open(self._fixture_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        self._fixture_data = rows
        logger.debug("Loaded %d rows from fixture %s", len(rows), self._fixture_path)
        return rows

    def _observations_from_fixture(
        self,
        latitude: float,
        longitude: float,
        start_dt: datetime,
        end_dt: datetime,
    ) -> List[RawObservation]:
        """Return observations from the CSV fixture nearest to the requested coords."""
        rows = self._load_fixture()
        loc_id = _make_loc_id(latitude, longitude)

        # Find the closest location in the fixture by lat/lon distance
        available_locs: dict = {}
        for row in rows:
            rid = row.get("location_id", "")
            if rid not in available_locs:
                try:
                    available_locs[rid] = (float(row["latitude"]), float(row["longitude"]))
                except (KeyError, ValueError):
                    pass

        if not available_locs:
            logger.warning("Fixture has no usable location data; using generator")
            return self._generate_observations(latitude, longitude, start_dt, end_dt)

        # Pick nearest fixture location
        best_loc_id = min(
            available_locs,
            key=lambda rid: _haversine(
                latitude, longitude,
                available_locs[rid][0], available_locs[rid][1],
            ),
        )
        best_lat, best_lon = available_locs[best_loc_id]

        result: List[RawObservation] = []
        for row in rows:
            if row.get("location_id", "") != best_loc_id:
                continue
            try:
                ts_dt = _parse_utc(row["timestamp_utc"])
            except Exception:
                continue
            if not (start_dt <= ts_dt <= end_dt):
                continue

            rain_raw = row.get("rainfall_mm", "").strip()
            rainfall_mm: Optional[float] = None
            if rain_raw and rain_raw.lower() not in ("", "nan", "null", "none"):
                try:
                    rainfall_mm = float(rain_raw)
                except ValueError:
                    pass

            result.append(
                RawObservation(
                    location_id=loc_id,  # use the REQUESTED location_id
                    latitude=latitude,
                    longitude=longitude,
                    timestamp_utc=ts_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    rainfall_mm=rainfall_mm,
                    source="mock_fixture",
                )
            )

        result.sort(key=lambda r: r.timestamp_utc)
        return result

    def _generate_observations(
        self,
        latitude: float,
        longitude: float,
        start_dt: datetime,
        end_dt: datetime,
    ) -> List[RawObservation]:
        """Generate deterministic synthetic observations."""
        loc_id = _make_loc_id(latitude, longitude)
        seed = int(abs(latitude * 1000 + longitude * 100)) % (2**31)
        rng = random.Random(seed)
        result: List[RawObservation] = []
        current = start_dt
        hour_idx = 0
        while current <= end_dt:
            ts_str = current.strftime("%Y-%m-%dT%H:%M:%SZ")
            # Sinusoidal base pattern (mm/hr)
            base = 3.0 + 2.5 * math.sin(hour_idx * math.pi / 12)
            noise = rng.uniform(0.0, 1.5)
            val = max(0.0, base + noise)
            result.append(
                RawObservation(
                    location_id=loc_id,
                    latitude=latitude,
                    longitude=longitude,
                    timestamp_utc=ts_str,
                    rainfall_mm=round(val, 2),
                    source="mock_generated",
                )
            )
            current += timedelta(hours=1)
            hour_idx += 1
        return result


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _parse_utc(ts: str) -> datetime:
    """Parse an ISO 8601 UTC string to a timezone-aware datetime."""
    ts = ts.strip()
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _make_loc_id(lat: float, lon: float) -> str:
    """Generate location_id compatible with Engineer 1's format."""
    return f"LOC_{lat:+.4f}_{lon:+08.4f}"


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return approximate great-circle distance in km."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
