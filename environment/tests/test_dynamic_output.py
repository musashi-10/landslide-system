"""
environment/tests/test_dynamic_output.py

Contract compliance tests.

Verifies that:
1. DynamicRecord contains all required data-contract fields.
2. Output values respect units (mm, non-negative rainfall).
3. Missing values are None, not 0.
4. DynamicRecord is mergeable into shared EnvironmentalFeatures schema.
5. No forbidden risk-prediction fields are present.
"""

from __future__ import annotations

import pytest

from environment.aggregation.feature_builder import build_dynamic_features, dynamic_record_to_dict
from environment.caching.local_cache import LocalCache
from environment.providers.mock_provider import MockWeatherProvider
from shared.schemas import EnvironmentalFeatures


REFERENCE_TIME = "2026-09-03T04:00:00Z"
LAT, LON = 27.123, 88.456
LOC_ID = "LOC_+27.1230_+088.4560"


def get_record():
    return build_dynamic_features(
        location_id=LOC_ID,
        latitude=LAT,
        longitude=LON,
        timestamp_utc=REFERENCE_TIME,
        provider=MockWeatherProvider(),
        cache=LocalCache(ttl_seconds=0),
    )


class TestDataContractCompliance:
    """Verify output matches docs/data-contract.md §6 dynamic features."""

    def test_required_spatial_fields_present(self):
        r = get_record()
        assert r.location_id is not None
        assert r.latitude is not None
        assert r.longitude is not None
        assert r.timestamp_utc is not None

    def test_rainfall_1h_field_present(self):
        r = get_record()
        assert hasattr(r, "rainfall_1h")

    def test_rainfall_6h_field_present(self):
        r = get_record()
        assert hasattr(r, "rainfall_6h")

    def test_rainfall_24h_field_present(self):
        r = get_record()
        assert hasattr(r, "rainfall_24h")

    def test_rainfall_3d_field_present(self):
        r = get_record()
        assert hasattr(r, "rainfall_3d")

    def test_rainfall_7d_field_present(self):
        r = get_record()
        assert hasattr(r, "rainfall_7d")

    def test_forecast_rainfall_field_present(self):
        r = get_record()
        assert hasattr(r, "forecast_rainfall")

    def test_moisture_indicator_field_present(self):
        r = get_record()
        assert hasattr(r, "moisture_indicator")

    def test_rainfall_units_non_negative(self):
        """All rainfall values must be >= 0 (unit: mm)."""
        r = get_record()
        for field in ("rainfall_1h", "rainfall_6h", "rainfall_24h", "rainfall_3d", "rainfall_7d"):
            val = getattr(r, field)
            if val is not None:
                assert val >= 0, f"{field} = {val} < 0 (invalid mm value)"

    def test_moisture_indicator_bounded(self):
        """moisture_indicator must be in [0, 1]."""
        r = get_record()
        if r.moisture_indicator is not None:
            assert 0.0 <= r.moisture_indicator <= 1.0

    def test_no_risk_prediction_fields(self):
        """Engineer 4 output must NOT contain risk/ML fields."""
        r = get_record()
        forbidden = ["landslide_probability", "risk_score", "risk_level", "risk_probability",
                     "warning_level", "alert_level"]
        for attr in forbidden:
            assert not hasattr(r, attr), f"Forbidden field found: {attr}"

    def test_timestamp_is_utc_format(self):
        """timestamp_utc must be ISO 8601 UTC (ends with Z)."""
        r = get_record()
        assert r.timestamp_utc.endswith("Z"), f"timestamp_utc does not end with Z: {r.timestamp_utc}"

    def test_location_id_format(self):
        """location_id must follow LOC_ format."""
        r = get_record()
        assert r.location_id.startswith("LOC_")


class TestEnvironmentalFeaturesIntegration:
    """Test that DynamicRecord can be merged into shared EnvironmentalFeatures."""

    def test_merge_into_environmental_features(self):
        """Engineer 5 should be able to merge dynamic features with static features."""
        r = get_record()
        ef = EnvironmentalFeatures(
            location_id=r.location_id,
            latitude=r.latitude,
            longitude=r.longitude,
            timestamp_utc=r.timestamp_utc,
            # Static features (would come from Engineer 2)
            elevation=1240.5,
            slope=32.4,
            aspect=145.0,
            soil_type="clay",
            geology="granite",
            land_cover="forest",
            historical_landslide=1,
            # Dynamic features (from Engineer 4)
            rainfall_1h=r.rainfall_1h,
            rainfall_6h=r.rainfall_6h,
            rainfall_24h=r.rainfall_24h,
            rainfall_3d=r.rainfall_3d,
            rainfall_7d=r.rainfall_7d,
            forecast_rainfall=r.forecast_rainfall,
            moisture_indicator=r.moisture_indicator,
        )
        assert ef.location_id == LOC_ID
        assert ef.rainfall_24h == r.rainfall_24h
        assert ef.moisture_indicator == r.moisture_indicator

    def test_missing_dynamic_fields_allowed(self):
        """EnvironmentalFeatures allows None for all dynamic fields."""
        ef = EnvironmentalFeatures(
            location_id=LOC_ID,
            latitude=LAT,
            longitude=LON,
        )
        # All dynamic fields should default to None — not zero
        assert ef.rainfall_1h is None
        assert ef.rainfall_6h is None
        assert ef.rainfall_24h is None
        assert ef.moisture_indicator is None


class TestJsonOutput:
    """Test that the dict output is JSON-serialisable and correct shape."""

    def test_dynamic_record_to_dict_is_json_compatible(self):
        import json
        r = get_record()
        d = dynamic_record_to_dict(r)
        # Should not raise
        json_str = json.dumps(d, default=str)
        assert json_str

    def test_dynamic_record_to_dict_matches_contract(self):
        """Example from data-contract.md §7 (dynamic fields only)."""
        r = get_record()
        d = dynamic_record_to_dict(r)
        contract_fields = [
            "location_id", "timestamp_utc",
            "rainfall_1h", "rainfall_6h", "rainfall_24h",
            "rainfall_3d", "rainfall_7d", "forecast_rainfall", "moisture_indicator",
        ]
        for field in contract_fields:
            assert field in d, f"Missing contract field: {field}"
