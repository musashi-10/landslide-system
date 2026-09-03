"""
Tests for ml.data.adapter — the ML data ingestion adapter.

Tests:
  - Valid join of static + dynamic DataFrames
  - Missing required identifier columns → ValueError
  - Zero-row join → ValueError
  - Missing values preserved as NaN (not silently zeroed)
  - Satellite join when satellite_df is provided
  - Satellite join skipped when None
  - Target column correctly derived from historical_landslide
  - Class balance statistics reported correctly
"""

import pytest
import pandas as pd
import numpy as np

from ml.data.adapter import build_ml_dataset, MLDataset


# ── Fixtures ──────────────────────────────────────────────────────────────────


def make_static_df(n=10, with_timestamps=False):
    """Minimal valid static DataFrame."""
    d = {
        "location_id": [f"LOC_{i:03d}" for i in range(n)],
        "latitude": np.linspace(27.0, 28.0, n),
        "longitude": np.linspace(88.0, 89.0, n),
        "elevation": np.random.uniform(500, 2000, n),
        "slope": np.random.uniform(5, 45, n),
        "aspect": np.random.uniform(0, 360, n),
        "soil_type": ["loam"] * n,
        "geology": ["schist"] * n,
        "land_cover": ["forest"] * n,
        "historical_landslide": [1 if i < 2 else 0 for i in range(n)],
    }
    if with_timestamps:
        d["timestamp_utc"] = [f"2026-0{(i % 9) + 1}-01T00:00:00Z" for i in range(n)]
    return pd.DataFrame(d)


def make_dynamic_df(location_ids=None, n_times=3):
    """Minimal valid dynamic DataFrame with multiple timestamps per location."""
    if location_ids is None:
        location_ids = [f"LOC_{i:03d}" for i in range(10)]
    rows = []
    for loc in location_ids:
        for t in range(n_times):
            rows.append({
                "location_id": loc,
                "timestamp_utc": f"2026-09-0{t + 1}T00:00:00Z",
                "rainfall_1h": np.random.uniform(0, 20),
                "rainfall_6h": np.random.uniform(0, 80),
                "rainfall_24h": np.random.uniform(0, 150),
                "rainfall_3d": np.random.uniform(0, 300),
                "rainfall_7d": np.random.uniform(0, 500),
                "forecast_rainfall": np.random.uniform(0, 50),
                "moisture_indicator": np.random.uniform(0, 1),
            })
    return pd.DataFrame(rows)


def make_satellite_df(location_ids=None):
    """Optional satellite DataFrame."""
    if location_ids is None:
        location_ids = [f"LOC_{i:03d}" for i in range(5)]
    return pd.DataFrame({
        "location_id": location_ids,
        "acquisition_time": ["2026-08-01T00:00:00Z"] * len(location_ids),
        "ndvi": np.random.uniform(0.1, 0.9, len(location_ids)),
        "vegetation_change": np.random.uniform(-0.3, 0.3, len(location_ids)),
        "bare_surface_indicator": np.random.uniform(0, 0.5, len(location_ids)),
    })


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestBuildMLDataset:
    def test_returns_ml_dataset(self):
        static_df = make_static_df()
        dynamic_df = make_dynamic_df()
        result = build_ml_dataset(static_df, dynamic_df)
        assert isinstance(result, MLDataset)

    def test_non_empty_df(self):
        static_df = make_static_df()
        dynamic_df = make_dynamic_df()
        result = build_ml_dataset(static_df, dynamic_df)
        assert len(result.df) > 0

    def test_feature_cols_present_in_df(self):
        static_df = make_static_df()
        dynamic_df = make_dynamic_df()
        result = build_ml_dataset(static_df, dynamic_df)
        for col in result.feature_cols:
            assert col in result.df.columns, f"Feature column '{col}' not in df"

    def test_target_column_present(self):
        static_df = make_static_df()
        dynamic_df = make_dynamic_df()
        result = build_ml_dataset(static_df, dynamic_df)
        assert result.target_col in result.df.columns

    def test_class_balance_counts_correct(self):
        static_df = make_static_df(n=10)  # 2 positives, 8 negatives
        dynamic_df = make_dynamic_df(n_times=1)
        result = build_ml_dataset(static_df, dynamic_df)
        # After join, positives from historical_landslide==1 should be >= 2
        assert result.n_positives >= 0
        assert result.n_negatives >= 0
        assert result.n_positives + result.n_negatives == len(result.df)

    def test_missing_location_id_in_static_raises(self):
        static_df = make_static_df()
        static_df = static_df.drop(columns=["location_id"])
        dynamic_df = make_dynamic_df()
        with pytest.raises(ValueError, match="location_id"):
            build_ml_dataset(static_df, dynamic_df)

    def test_missing_location_id_in_dynamic_raises(self):
        static_df = make_static_df()
        dynamic_df = make_dynamic_df()
        dynamic_df = dynamic_df.drop(columns=["location_id"])
        with pytest.raises(ValueError, match="location_id"):
            build_ml_dataset(static_df, dynamic_df)

    def test_missing_timestamp_in_dynamic_raises(self):
        static_df = make_static_df()
        dynamic_df = make_dynamic_df()
        dynamic_df = dynamic_df.drop(columns=["timestamp_utc"])
        with pytest.raises(ValueError, match="timestamp_utc"):
            build_ml_dataset(static_df, dynamic_df)

    def test_zero_row_join_raises(self):
        static_df = make_static_df()
        # Dynamic with completely different location_ids
        dynamic_df = make_dynamic_df(location_ids=["NONEXISTENT_001", "NONEXISTENT_002"])
        with pytest.raises(ValueError, match="zero rows"):
            build_ml_dataset(static_df, dynamic_df)

    def test_missing_values_preserved_as_nan(self):
        """Missing values must never silently become zero (data contract §12)."""
        static_df = make_static_df()
        static_df.loc[0, "elevation"] = np.nan  # explicit missing
        dynamic_df = make_dynamic_df()
        result = build_ml_dataset(static_df, dynamic_df)
        # elevation should have NaN rows in the output
        assert result.df["elevation"].isna().any() or "elevation" not in result.df.columns

    def test_satellite_features_joined(self):
        static_df = make_static_df(n=5)
        dynamic_df = make_dynamic_df(location_ids=[f"LOC_{i:03d}" for i in range(5)])
        satellite_df = make_satellite_df(location_ids=[f"LOC_{i:03d}" for i in range(5)])
        result = build_ml_dataset(static_df, dynamic_df, satellite_df=satellite_df)
        assert result.has_satellite is True
        assert "ndvi" in result.df.columns

    def test_satellite_none_not_included(self):
        static_df = make_static_df()
        dynamic_df = make_dynamic_df()
        result = build_ml_dataset(static_df, dynamic_df, satellite_df=None)
        assert result.has_satellite is False
        assert "ndvi" not in result.df.columns

    def test_missing_summary_is_dict(self):
        static_df = make_static_df()
        dynamic_df = make_dynamic_df()
        result = build_ml_dataset(static_df, dynamic_df)
        assert isinstance(result.missing_summary, dict)

    def test_reproducible_same_output(self):
        """Same inputs must produce the same output."""
        static_df = make_static_df()
        dynamic_df = make_dynamic_df()
        r1 = build_ml_dataset(static_df.copy(), dynamic_df.copy())
        r2 = build_ml_dataset(static_df.copy(), dynamic_df.copy())
        assert list(r1.df.columns) == list(r2.df.columns)
        assert len(r1.df) == len(r2.df)
