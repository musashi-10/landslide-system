"""
environment/tests/test_forecast_features.py

Unit tests for forecast rainfall feature computation.

Critical correctness requirement:
  missing forecast != 0 rainfall

Tests cover:
- Normal forecast accumulation
- Empty forecast list -> None (NOT 0)
- All-None forecast records -> None (NOT 0)
- Partial None records -> partial sum with warning
- horizon_hours filtering
- align_forecast_to_horizon function
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta

import pytest

from environment.features.forecast_features import (
    align_forecast_to_horizon,
    calculate_forecast_rainfall,
)
from environment.schemas.dynamic_record import ForecastRecord


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_forecast(hours: int = 24, value_mm: float = 5.0, loc: str = "LOC_001") -> list:
    """Create a list of hourly ForecastRecord with constant rainfall."""
    base = datetime(2026, 9, 2, 13, 0, 0, tzinfo=timezone.utc)
    records = []
    for h in range(hours):
        ts = (base + timedelta(hours=h)).strftime("%Y-%m-%dT%H:%M:%SZ")
        records.append(
            ForecastRecord(
                location_id=loc,
                latitude=27.123,
                longitude=88.456,
                timestamp_utc=ts,
                forecast_rainfall_mm=value_mm,
                source="mock",
            )
        )
    return records


def make_forecast_with_none(hours: int = 24) -> list:
    """Create forecast records where every value is None."""
    base = datetime(2026, 9, 2, 13, 0, 0, tzinfo=timezone.utc)
    records = []
    for h in range(hours):
        ts = (base + timedelta(hours=h)).strftime("%Y-%m-%dT%H:%M:%SZ")
        records.append(
            ForecastRecord(
                location_id="LOC_001",
                latitude=27.123,
                longitude=88.456,
                timestamp_utc=ts,
                forecast_rainfall_mm=None,  # missing
                source="mock",
            )
        )
    return records


# ---------------------------------------------------------------------------
# Tests: calculate_forecast_rainfall
# ---------------------------------------------------------------------------


class TestCalculateForecastRainfall:
    def test_empty_list_returns_none_not_zero(self):
        """CRITICAL: empty forecast must return None, not 0."""
        result = calculate_forecast_rainfall([])
        assert result is None, f"Expected None for empty forecast, got {result}"

    def test_all_none_values_returns_none_not_zero(self):
        """CRITICAL: all-None forecast records must return None, not 0."""
        records = make_forecast_with_none(hours=24)
        result = calculate_forecast_rainfall(records, horizon_hours=24)
        assert result is None, f"Expected None when all forecast values are None, got {result}"

    def test_basic_accumulation(self):
        records = make_forecast(hours=24, value_mm=2.0)
        result = calculate_forecast_rainfall(records, horizon_hours=24)
        assert result == pytest.approx(48.0, abs=0.1)

    def test_horizon_truncation(self):
        """Only first horizon_hours records should be summed."""
        records = make_forecast(hours=48, value_mm=1.0)
        result_6h = calculate_forecast_rainfall(records, horizon_hours=6)
        result_24h = calculate_forecast_rainfall(records, horizon_hours=24)
        assert result_6h == pytest.approx(6.0, abs=0.1)
        assert result_24h == pytest.approx(24.0, abs=0.1)

    def test_partial_none_returns_partial_sum(self):
        """Partial missing -> sum available, log warning."""
        records = make_forecast(hours=24, value_mm=5.0)
        # Set 4 records to None
        for i in range(0, 4):
            records[i] = ForecastRecord(
                location_id=records[i].location_id,
                latitude=records[i].latitude,
                longitude=records[i].longitude,
                timestamp_utc=records[i].timestamp_utc,
                forecast_rainfall_mm=None,
                source="mock",
            )
        result = calculate_forecast_rainfall(records, horizon_hours=24)
        # 20 valid * 5.0 = 100
        assert result is not None
        assert result == pytest.approx(100.0, abs=0.1)

    def test_non_negative_result(self):
        records = make_forecast(hours=12, value_mm=0.0)
        result = calculate_forecast_rainfall(records, horizon_hours=12)
        assert result is not None
        assert result == pytest.approx(0.0, abs=0.01)

    def test_single_record(self):
        records = make_forecast(hours=1, value_mm=15.0)
        result = calculate_forecast_rainfall(records, horizon_hours=24)
        assert result == pytest.approx(15.0, abs=0.01)


# ---------------------------------------------------------------------------
# Tests: align_forecast_to_horizon
# ---------------------------------------------------------------------------


class TestAlignForecastToHorizon:
    def test_filters_past_records(self):
        """Records before reference_utc should be excluded."""
        now = datetime(2026, 9, 2, 12, 0, 0, tzinfo=timezone.utc)
        ref_str = now.strftime("%Y-%m-%dT%H:%M:%SZ")

        records = []
        for h in range(-2, 25):  # from 2 hours past to 24 hours future
            ts = (now + timedelta(hours=h)).strftime("%Y-%m-%dT%H:%M:%SZ")
            records.append(
                ForecastRecord(
                    location_id="LOC_001",
                    latitude=0.0,
                    longitude=0.0,
                    timestamp_utc=ts,
                    forecast_rainfall_mm=1.0,
                )
            )

        aligned = align_forecast_to_horizon(records, ref_str, horizon_hours=24)
        # All returned timestamps must be > reference_utc
        for rec in aligned:
            assert rec.timestamp_utc > ref_str

    def test_filters_beyond_horizon(self):
        now = datetime(2026, 9, 2, 12, 0, 0, tzinfo=timezone.utc)
        ref_str = now.strftime("%Y-%m-%dT%H:%M:%SZ")
        records = make_forecast(hours=48, value_mm=1.0)
        # Overwrite timestamps to be relative to now
        updated = []
        for i, rec in enumerate(records):
            ts = (now + timedelta(hours=i + 1)).strftime("%Y-%m-%dT%H:%M:%SZ")
            updated.append(
                ForecastRecord(
                    location_id=rec.location_id,
                    latitude=rec.latitude,
                    longitude=rec.longitude,
                    timestamp_utc=ts,
                    forecast_rainfall_mm=rec.forecast_rainfall_mm,
                )
            )
        aligned = align_forecast_to_horizon(updated, ref_str, horizon_hours=24)
        assert len(aligned) <= 24

    def test_empty_input_returns_empty(self):
        result = align_forecast_to_horizon([], "2026-09-02T12:00:00Z", horizon_hours=24)
        assert result == []

    def test_invalid_reference_returns_empty(self):
        records = make_forecast(hours=5)
        result = align_forecast_to_horizon(records, "not-a-timestamp", horizon_hours=24)
        assert result == []
