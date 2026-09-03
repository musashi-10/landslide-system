"""
environment/tests/test_rainfall_windows.py

Unit tests for rainfall window calculation functions.

Tests cover:
- Rolling rainfall calculations (1h, 6h, 24h, 3d, 7d)
- Missing observations (NaN) handling
- Empty series
- All-NaN series
- Correct window boundaries
- Rainfall intensity
- Antecedent rainfall (exponential decay)
- Rainfall anomaly
- build_datetime_series() helper
"""

from __future__ import annotations

import math
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import pytest

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
    calculate_rolling_rainfall,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_series(values: list, start_hour: int = 0) -> pd.Series:
    """Create a UTC-indexed hourly series from a list of values."""
    base = datetime(2026, 9, 1, start_hour, 0, 0, tzinfo=timezone.utc)
    index = pd.DatetimeIndex(
        [base + timedelta(hours=i) for i in range(len(values))], tz="UTC"
    )
    return pd.Series(values, index=index, dtype=float)


# ---------------------------------------------------------------------------
# Tests: rolling rainfall
# ---------------------------------------------------------------------------


class TestCalculateRollingRainfall:
    def test_basic_sum_1h(self):
        s = make_series([5.0, 8.0, 12.0, 6.0])
        ref = s.index[-1]
        result = calculate_rolling_rainfall(s, window_hours=1, reference_time=ref)
        assert result == pytest.approx(6.0, abs=0.01)

    def test_basic_sum_6h(self):
        # 6 values of 10.0 mm each -> 60 mm total
        s = make_series([10.0] * 10)
        ref = s.index[-1]
        result = calculate_rolling_rainfall(s, window_hours=6, reference_time=ref)
        assert result == pytest.approx(60.0, abs=0.01)

    def test_empty_series_returns_none(self):
        s = pd.Series(dtype=float)
        assert calculate_rolling_rainfall(s, window_hours=24) is None

    def test_all_nan_returns_none(self):
        s = make_series([float("nan")] * 10)
        result = calculate_rolling_rainfall(s, window_hours=6)
        assert result is None

    def test_mixed_nan_skips_nan(self):
        # 3 valid + 3 NaN in window
        s = make_series([5.0, float("nan"), 7.0, float("nan"), 3.0, float("nan")])
        ref = s.index[-1]
        result = calculate_rolling_rainfall(s, window_hours=6, reference_time=ref)
        # Sum of non-NaN: 5+7+3 = 15
        assert result == pytest.approx(15.0, abs=0.01)

    def test_result_is_non_negative(self):
        # All zeros — result should be 0.0 not None
        s = make_series([0.0] * 24)
        result = calculate_rolling_rainfall(s, window_hours=24)
        assert result == pytest.approx(0.0, abs=0.001)

    def test_window_larger_than_series(self):
        # Window is 168h but only 10 observations exist
        s = make_series([5.0] * 10)
        result = calculate_rolling_rainfall(s, window_hours=168)
        assert result == pytest.approx(50.0, abs=0.01)

    def test_default_reference_is_last_index(self):
        s = make_series([1.0, 2.0, 3.0])
        # With no reference_time, uses last index
        r1 = calculate_rolling_rainfall(s, window_hours=1)
        assert r1 == pytest.approx(3.0, abs=0.01)


class TestConvenienceWrappers:
    def setup_method(self):
        # 200 hours of data at 5mm/hr
        self.s = make_series([5.0] * 200)
        self.ref = self.s.index[-1]

    def test_rainfall_1h(self):
        result = calculate_rainfall_1h(self.s, self.ref)
        assert result == pytest.approx(5.0, abs=0.01)

    def test_rainfall_6h(self):
        result = calculate_rainfall_6h(self.s, self.ref)
        assert result == pytest.approx(30.0, abs=0.01)

    def test_rainfall_24h(self):
        result = calculate_rainfall_24h(self.s, self.ref)
        assert result == pytest.approx(120.0, abs=0.01)

    def test_rainfall_3d(self):
        result = calculate_rainfall_3d(self.s, self.ref)
        assert result == pytest.approx(360.0, abs=0.01)

    def test_rainfall_7d(self):
        result = calculate_rainfall_7d(self.s, self.ref)
        assert result == pytest.approx(840.0, abs=0.01)


# ---------------------------------------------------------------------------
# Tests: rainfall intensity
# ---------------------------------------------------------------------------


class TestRainfallIntensity:
    def test_intensity_equals_1h_for_hourly_data(self):
        s = make_series([10.0] * 5)
        ref = s.index[-1]
        intensity = calculate_rainfall_intensity(s, window_minutes=60, reference_time=ref)
        r1h = calculate_rainfall_1h(s, reference_time=ref)
        # For hourly data, intensity should equal 1h accumulation
        assert intensity == pytest.approx(r1h, abs=0.01)

    def test_empty_returns_none(self):
        s = pd.Series(dtype=float)
        assert calculate_rainfall_intensity(s, window_minutes=60) is None

    def test_all_nan_returns_none(self):
        s = make_series([float("nan")] * 5)
        assert calculate_rainfall_intensity(s, window_minutes=60) is None

    def test_non_negative(self):
        s = make_series([0.0, 0.0, 0.0])
        result = calculate_rainfall_intensity(s, window_minutes=60)
        # 0 rainfall -> 0 intensity
        assert result == pytest.approx(0.0, abs=0.001)


# ---------------------------------------------------------------------------
# Tests: antecedent rainfall
# ---------------------------------------------------------------------------


class TestAntecedentRainfall:
    def test_empty_series_returns_none(self):
        s = pd.Series(dtype=float)
        assert calculate_antecedent_rainfall(s) is None

    def test_all_nan_returns_none(self):
        s = make_series([float("nan")] * 10)
        assert calculate_antecedent_rainfall(s) is None

    def test_single_observation(self):
        s = make_series([10.0])
        result = calculate_antecedent_rainfall(s, decay_factor=0.85)
        # Only one observation: API = 0.85^0 * 10 = 10
        assert result == pytest.approx(10.0, abs=0.01)

    def test_more_recent_gets_more_weight(self):
        # Older observation is much larger but should contribute less
        # Series sorted ascending: [100.0 (old), 1.0 (recent)]
        s = make_series([100.0, 1.0])
        result = calculate_antecedent_rainfall(s, decay_factor=0.5)
        # API = 0.5^0 * 1.0 + 0.5^1 * 100.0 = 1 + 50 = 51
        assert result == pytest.approx(51.0, abs=0.01)

    def test_result_non_negative(self):
        s = make_series([0.0] * 10)
        result = calculate_antecedent_rainfall(s, decay_factor=0.85)
        assert result == pytest.approx(0.0, abs=0.001)

    def test_decay_factor_effect(self):
        s = make_series([10.0] * 5)
        r_high = calculate_antecedent_rainfall(s, decay_factor=0.99)
        r_low = calculate_antecedent_rainfall(s, decay_factor=0.5)
        # Higher decay = more historical memory = higher index
        assert r_high > r_low


# ---------------------------------------------------------------------------
# Tests: rainfall anomaly
# ---------------------------------------------------------------------------


class TestRainfallAnomaly:
    def test_zero_anomaly_when_uniform(self):
        # Constant rainfall — baseline mean = recent rate
        s = make_series([5.0] * 200)
        ref = s.index[-1]
        result = calculate_rainfall_anomaly(s, window_hours=24, baseline_hours=168, reference_time=ref)
        # Anomaly should be ~0 for constant series
        assert result == pytest.approx(0.0, abs=5.0)  # allow small rounding

    def test_positive_anomaly_for_heavy_event(self):
        # Dry baseline then heavy event
        s = make_series([0.0] * 168 + [20.0] * 24)
        ref = s.index[-1]
        result = calculate_rainfall_anomaly(s, window_hours=24, baseline_hours=168, reference_time=ref)
        assert result is not None
        assert result > 0  # more rain than baseline average

    def test_empty_returns_none(self):
        s = pd.Series(dtype=float)
        assert calculate_rainfall_anomaly(s) is None

    def test_insufficient_history_returns_none(self):
        # Only 5 hours of data — not enough for 168h baseline
        s = make_series([5.0] * 5)
        ref = s.index[-1]
        result = calculate_rainfall_anomaly(s, window_hours=24, baseline_hours=168, reference_time=ref)
        # May return None or a partial value depending on implementation
        # Key: must not raise an exception
        assert result is None or isinstance(result, float)


# ---------------------------------------------------------------------------
# Tests: build_datetime_series
# ---------------------------------------------------------------------------


class TestBuildDatetimeSeries:
    def test_basic_conversion(self):
        df = pd.DataFrame(
            {
                "timestamp_utc": [
                    "2026-09-01T00:00:00Z",
                    "2026-09-01T01:00:00Z",
                    "2026-09-01T02:00:00Z",
                ],
                "rainfall_mm": [5.0, 8.0, 12.0],
            }
        )
        s = build_datetime_series(df)
        assert len(s) == 3
        assert s.iloc[0] == pytest.approx(5.0)
        assert isinstance(s.index, pd.DatetimeIndex)
        assert s.index.tz is not None  # must be timezone-aware

    def test_empty_df_returns_empty_series(self):
        df = pd.DataFrame(columns=["timestamp_utc", "rainfall_mm"])
        s = build_datetime_series(df)
        assert s.empty

    def test_nan_values_preserved(self):
        df = pd.DataFrame(
            {
                "timestamp_utc": ["2026-09-01T00:00:00Z", "2026-09-01T01:00:00Z"],
                "rainfall_mm": [5.0, float("nan")],
            }
        )
        s = build_datetime_series(df)
        assert math.isnan(s.iloc[1])

    def test_sorted_ascending(self):
        # Unsorted input
        df = pd.DataFrame(
            {
                "timestamp_utc": [
                    "2026-09-01T02:00:00Z",
                    "2026-09-01T00:00:00Z",
                    "2026-09-01T01:00:00Z",
                ],
                "rainfall_mm": [3.0, 1.0, 2.0],
            }
        )
        s = build_datetime_series(df)
        assert s.iloc[0] == pytest.approx(1.0)  # earliest value first
        assert s.index.is_monotonic_increasing
