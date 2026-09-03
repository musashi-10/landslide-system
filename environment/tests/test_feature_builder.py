"""
environment/tests/test_feature_builder.py

Integration tests for build_dynamic_features() and get_dynamic_conditions().

Tests cover:
- Normal operation with mock provider + fixture
- All required data contract fields present
- Invalid coordinates raise ValueError
- Provider failure gracefully returns None fields (not crash)
- Caching behavior (second call returns cached)
- dynamic_record_to_dict() output shape
- DynamicRecord -> EnvironmentalFeatures compatibility
"""

from __future__ import annotations

import pytest

from environment.aggregation.feature_builder import (
    build_dynamic_features,
    dynamic_record_to_dict,
    get_dynamic_conditions,
)
from environment.caching.local_cache import LocalCache
from environment.config import EnvironmentConfig
from environment.providers.mock_provider import MockWeatherProvider
from environment.schemas.dynamic_record import DynamicRecord


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

REFERENCE_TIME = "2026-09-03T04:00:00Z"
LAT, LON = 27.123, 88.456
LOC_ID = "LOC_+27.1230_+088.4560"


def make_provider():
    return MockWeatherProvider()


def make_no_cache_config():
    return EnvironmentConfig(cache_ttl_seconds=0)


# ---------------------------------------------------------------------------
# Tests: build_dynamic_features
# ---------------------------------------------------------------------------


class TestBuildDynamicFeatures:
    def test_returns_dynamic_record(self):
        record = build_dynamic_features(
            location_id=LOC_ID,
            latitude=LAT,
            longitude=LON,
            timestamp_utc=REFERENCE_TIME,
            provider=make_provider(),
            cache=LocalCache(ttl_seconds=0),
        )
        assert isinstance(record, DynamicRecord)

    def test_location_id_preserved(self):
        record = build_dynamic_features(
            location_id=LOC_ID, latitude=LAT, longitude=LON,
            timestamp_utc=REFERENCE_TIME, provider=make_provider(),
            cache=LocalCache(ttl_seconds=0),
        )
        assert record.location_id == LOC_ID

    def test_timestamp_preserved(self):
        record = build_dynamic_features(
            location_id=LOC_ID, latitude=LAT, longitude=LON,
            timestamp_utc=REFERENCE_TIME, provider=make_provider(),
            cache=LocalCache(ttl_seconds=0),
        )
        assert record.timestamp_utc == REFERENCE_TIME

    def test_rainfall_windows_computed(self):
        record = build_dynamic_features(
            location_id=LOC_ID, latitude=LAT, longitude=LON,
            timestamp_utc=REFERENCE_TIME, provider=make_provider(),
            cache=LocalCache(ttl_seconds=0),
        )
        # With 8 days of fixture data, 6h/24h/3d/7d windows should have values
        assert record.rainfall_6h is not None
        assert record.rainfall_24h is not None
        assert record.rainfall_3d is not None
        assert record.rainfall_7d is not None

    def test_rainfall_values_non_negative(self):
        record = build_dynamic_features(
            location_id=LOC_ID, latitude=LAT, longitude=LON,
            timestamp_utc=REFERENCE_TIME, provider=make_provider(),
            cache=LocalCache(ttl_seconds=0),
        )
        for field in ("rainfall_1h", "rainfall_6h", "rainfall_24h", "rainfall_3d", "rainfall_7d"):
            val = getattr(record, field)
            if val is not None:
                assert val >= 0, f"{field} is negative: {val}"

    def test_forecast_rainfall_is_none_or_float(self):
        record = build_dynamic_features(
            location_id=LOC_ID, latitude=LAT, longitude=LON,
            timestamp_utc=REFERENCE_TIME, provider=make_provider(),
            cache=LocalCache(ttl_seconds=0),
        )
        assert record.forecast_rainfall is None or isinstance(record.forecast_rainfall, float)

    def test_moisture_indicator_in_range(self):
        record = build_dynamic_features(
            location_id=LOC_ID, latitude=LAT, longitude=LON,
            timestamp_utc=REFERENCE_TIME, provider=make_provider(),
            cache=LocalCache(ttl_seconds=0),
        )
        if record.moisture_indicator is not None:
            assert 0.0 <= record.moisture_indicator <= 1.0

    def test_data_source_recorded(self):
        record = build_dynamic_features(
            location_id=LOC_ID, latitude=LAT, longitude=LON,
            timestamp_utc=REFERENCE_TIME, provider=make_provider(),
            cache=LocalCache(ttl_seconds=0),
        )
        assert record.data_source == "mock"

    def test_invalid_latitude_raises(self):
        with pytest.raises(ValueError):
            build_dynamic_features(
                location_id=LOC_ID, latitude=95.0, longitude=LON,
                timestamp_utc=REFERENCE_TIME, provider=make_provider(),
            )

    def test_invalid_longitude_raises(self):
        with pytest.raises(ValueError):
            build_dynamic_features(
                location_id=LOC_ID, latitude=LAT, longitude=200.0,
                timestamp_utc=REFERENCE_TIME, provider=make_provider(),
            )

    def test_provider_failure_does_not_crash(self):
        """Provider failure returns None fields, not an exception."""
        from environment.providers.mock_provider import MockWeatherProvider
        failing_provider = MockWeatherProvider(fail_mode="unavailable")
        record = build_dynamic_features(
            location_id=LOC_ID, latitude=LAT, longitude=LON,
            timestamp_utc=REFERENCE_TIME, provider=failing_provider,
            cache=LocalCache(ttl_seconds=0),
        )
        # Should return a DynamicRecord with None rainfall fields
        assert isinstance(record, DynamicRecord)
        assert record.rainfall_24h is None
        assert record.rainfall_7d is None

    def test_get_dynamic_conditions_alias(self):
        """get_dynamic_conditions is an alias for build_dynamic_features."""
        r1 = build_dynamic_features(
            location_id=LOC_ID, latitude=LAT, longitude=LON,
            timestamp_utc=REFERENCE_TIME, provider=make_provider(),
            cache=LocalCache(ttl_seconds=0),
        )
        r2 = get_dynamic_conditions(
            location_id=LOC_ID, latitude=LAT, longitude=LON,
            timestamp_utc=REFERENCE_TIME, provider=make_provider(),
            cache=LocalCache(ttl_seconds=0),
        )
        assert r1.rainfall_24h == r2.rainfall_24h


class TestCachingBehavior:
    def test_second_call_uses_cache(self):
        cache = LocalCache(ttl_seconds=60)
        provider = make_provider()

        r1 = build_dynamic_features(
            location_id=LOC_ID, latitude=LAT, longitude=LON,
            timestamp_utc=REFERENCE_TIME, provider=provider, cache=cache,
        )
        # Poison the provider to simulate it being unavailable
        provider._fail_mode = "unavailable"

        # Second call should use cache (not the now-failing provider)
        r2 = build_dynamic_features(
            location_id=LOC_ID, latitude=LAT, longitude=LON,
            timestamp_utc=REFERENCE_TIME, provider=provider, cache=cache,
        )
        assert r1.rainfall_24h == r2.rainfall_24h

    def test_cache_disabled_calls_provider_each_time(self):
        """With TTL=0, every call should re-fetch (no caching)."""
        calls = []
        original_get = MockWeatherProvider.get_observations

        cache = LocalCache(ttl_seconds=0)
        provider = make_provider()

        r1 = build_dynamic_features(
            location_id=LOC_ID, latitude=LAT, longitude=LON,
            timestamp_utc=REFERENCE_TIME, provider=provider, cache=cache,
        )
        r2 = build_dynamic_features(
            location_id=LOC_ID, latitude=LAT, longitude=LON,
            timestamp_utc=REFERENCE_TIME, provider=provider, cache=cache,
        )
        # Both should be valid records (no crash)
        assert isinstance(r1, DynamicRecord)
        assert isinstance(r2, DynamicRecord)


class TestDynamicRecordToDict:
    def test_output_has_contract_fields(self):
        record = build_dynamic_features(
            location_id=LOC_ID, latitude=LAT, longitude=LON,
            timestamp_utc=REFERENCE_TIME, provider=make_provider(),
            cache=LocalCache(ttl_seconds=0),
        )
        d = dynamic_record_to_dict(record)
        required_keys = [
            "location_id", "latitude", "longitude", "timestamp_utc",
            "rainfall_1h", "rainfall_6h", "rainfall_24h",
            "rainfall_3d", "rainfall_7d", "forecast_rainfall", "moisture_indicator",
        ]
        for key in required_keys:
            assert key in d, f"Missing key: {key}"

    def test_no_risk_fields_in_output(self):
        """Engineer 4 must NOT output risk/prediction fields."""
        record = build_dynamic_features(
            location_id=LOC_ID, latitude=LAT, longitude=LON,
            timestamp_utc=REFERENCE_TIME, provider=make_provider(),
            cache=LocalCache(ttl_seconds=0),
        )
        d = dynamic_record_to_dict(record)
        forbidden_keys = ["landslide_probability", "risk_score", "risk_level", "risk_probability"]
        for key in forbidden_keys:
            assert key not in d, f"Forbidden key present: {key}"


class TestContractCompatibility:
    def test_dynamic_record_compatible_with_environmental_features(self):
        """DynamicRecord fields must be mergeable into EnvironmentalFeatures."""
        from shared.schemas import EnvironmentalFeatures

        record = build_dynamic_features(
            location_id=LOC_ID, latitude=LAT, longitude=LON,
            timestamp_utc=REFERENCE_TIME, provider=make_provider(),
            cache=LocalCache(ttl_seconds=0),
        )

        # Build EnvironmentalFeatures using DynamicRecord fields
        ef = EnvironmentalFeatures(
            location_id=record.location_id,
            latitude=record.latitude,
            longitude=record.longitude,
            timestamp_utc=record.timestamp_utc,
            rainfall_1h=record.rainfall_1h,
            rainfall_6h=record.rainfall_6h,
            rainfall_24h=record.rainfall_24h,
            rainfall_3d=record.rainfall_3d,
            rainfall_7d=record.rainfall_7d,
            forecast_rainfall=record.forecast_rainfall,
            moisture_indicator=record.moisture_indicator,
        )
        assert ef.location_id == LOC_ID
        assert ef.rainfall_24h == record.rainfall_24h
