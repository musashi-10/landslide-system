"""
environment/tests/test_provider_mock.py

Tests for the MockWeatherProvider.

Covers:
- Normal observation fetch from fixture
- Normal observation fetch via generator (no fixture)
- Forecast generation
- Simulated failure modes (timeout, unavailable)
- Missing location handling
- Output schema validation
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest

from environment.providers.base import ProviderTimeoutError, ProviderUnavailableError
from environment.providers.mock_provider import MockWeatherProvider
from environment.schemas.dynamic_record import ForecastRecord, RawObservation

# Fixture path (relative to project root, resolved to absolute)
FIXTURE_PATH = Path(__file__).parent / "fixtures" / "synthetic_rainfall.csv"


class TestMockWeatherProviderObservations:
    def test_name(self):
        p = MockWeatherProvider()
        assert p.name == "mock"

    def test_returns_raw_observations(self):
        p = MockWeatherProvider(fixture_path=FIXTURE_PATH)
        obs = p.get_observations(
            latitude=27.123,
            longitude=88.456,
            start_utc="2026-08-27T00:00:00Z",
            end_utc="2026-08-27T05:00:00Z",
        )
        assert isinstance(obs, list)
        assert len(obs) > 0
        assert all(isinstance(o, RawObservation) for o in obs)

    def test_observations_have_valid_timestamps(self):
        p = MockWeatherProvider(fixture_path=FIXTURE_PATH)
        obs = p.get_observations(27.123, 88.456, "2026-08-27T00:00:00Z", "2026-08-27T05:00:00Z")
        for o in obs:
            assert o.timestamp_utc.endswith("Z")
            # Must be parseable
            ts = o.timestamp_utc[:-1] + "+00:00"
            datetime.fromisoformat(ts)

    def test_observations_sorted_chronologically(self):
        p = MockWeatherProvider(fixture_path=FIXTURE_PATH)
        obs = p.get_observations(27.123, 88.456, "2026-08-27T00:00:00Z", "2026-08-28T00:00:00Z")
        timestamps = [o.timestamp_utc for o in obs]
        assert timestamps == sorted(timestamps)

    def test_no_negative_rainfall(self):
        """Provider must never return negative rainfall values."""
        p = MockWeatherProvider(fixture_path=FIXTURE_PATH)
        obs = p.get_observations(27.123, 88.456, "2026-08-27T00:00:00Z", "2026-09-03T00:00:00Z")
        for o in obs:
            if o.rainfall_mm is not None:
                assert o.rainfall_mm >= 0, f"Negative rainfall: {o.rainfall_mm}"

    def test_missing_values_are_none_not_zero(self):
        """Missing obs must be None, not 0."""
        p = MockWeatherProvider(fixture_path=FIXTURE_PATH)
        obs = p.get_observations(27.123, 88.456, "2026-08-26T00:00:00Z", "2026-09-03T00:00:00Z")
        # Our fixture has one deliberate missing value (hour=10)
        none_count = sum(1 for o in obs if o.rainfall_mm is None)
        # At least one missing value per location in fixture
        assert none_count >= 1

    def test_generator_fallback_when_no_fixture(self):
        """Generator mode works when fixture does not exist."""
        fake_path = Path("/nonexistent/path/to/fixture.csv")
        p = MockWeatherProvider(fixture_path=fake_path)
        obs = p.get_observations(27.123, 88.456, "2026-09-01T00:00:00Z", "2026-09-01T05:00:00Z")
        assert len(obs) == 6  # 6 hours
        assert all(o.rainfall_mm is not None for o in obs)
        assert all(o.rainfall_mm >= 0 for o in obs)

    def test_location_id_format(self):
        """location_id must match Engineer 1's format."""
        p = MockWeatherProvider(fixture_path=FIXTURE_PATH)
        obs = p.get_observations(27.123, 88.456, "2026-08-27T00:00:00Z", "2026-08-27T02:00:00Z")
        if obs:
            assert obs[0].location_id.startswith("LOC_")


class TestMockWeatherProviderForecast:
    def test_returns_forecast_records(self):
        p = MockWeatherProvider()
        fcast = p.get_forecast(27.123, 88.456, horizon_hours=24)
        assert isinstance(fcast, list)
        assert len(fcast) == 24
        assert all(isinstance(f, ForecastRecord) for f in fcast)

    def test_forecast_sorted_chronologically(self):
        p = MockWeatherProvider()
        fcast = p.get_forecast(27.123, 88.456, horizon_hours=12)
        timestamps = [f.timestamp_utc for f in fcast]
        assert timestamps == sorted(timestamps)

    def test_forecast_non_negative(self):
        p = MockWeatherProvider()
        fcast = p.get_forecast(27.123, 88.456, horizon_hours=24)
        for f in fcast:
            if f.forecast_rainfall_mm is not None:
                assert f.forecast_rainfall_mm >= 0

    def test_forecast_horizon_respected(self):
        p = MockWeatherProvider()
        fcast6 = p.get_forecast(27.123, 88.456, horizon_hours=6)
        fcast24 = p.get_forecast(27.123, 88.456, horizon_hours=24)
        assert len(fcast6) == 6
        assert len(fcast24) == 24


class TestMockWeatherProviderFailModes:
    def test_timeout_raises_provider_timeout_error(self):
        p = MockWeatherProvider(fail_mode="timeout")
        with pytest.raises(ProviderTimeoutError):
            p.get_observations(27.123, 88.456, "2026-09-01T00:00:00Z", "2026-09-01T05:00:00Z")

    def test_unavailable_raises_provider_unavailable_error(self):
        p = MockWeatherProvider(fail_mode="unavailable")
        with pytest.raises(ProviderUnavailableError):
            p.get_observations(27.123, 88.456, "2026-09-01T00:00:00Z", "2026-09-01T05:00:00Z")

    def test_timeout_on_forecast_too(self):
        p = MockWeatherProvider(fail_mode="timeout")
        with pytest.raises(ProviderTimeoutError):
            p.get_forecast(27.123, 88.456, horizon_hours=24)


class TestMockProvenance:
    def test_provenance_has_required_fields(self):
        p = MockWeatherProvider()
        prov = p.provenance
        assert prov.source == "mock"
        assert prov.spatial_resolution != ""
        assert prov.temporal_resolution != ""
        assert prov.rainfall_value_type != ""
        assert "mm" in prov.units.lower()
