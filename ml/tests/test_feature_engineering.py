"""
Tests for ml.features.engineering — feature engineering module.

Tests:
  - All engineered columns are added
  - Same input → same output (reproducibility)
  - No NaN leakage (NaN inputs produce NaN outputs, not zeros)
  - Missing input columns produce NaN (not crashes)
  - engineer_features() convenience function works
  - engineered_column_names() returns correct list
"""

import numpy as np
import pandas as pd
import pytest

from ml.features.engineering import FeatureEngineer, engineer_features


def make_full_df(n=20):
    """DataFrame with all expected input columns."""
    np.random.seed(42)
    return pd.DataFrame({
        "location_id": [f"LOC_{i:03d}" for i in range(n)],
        "elevation": np.random.uniform(500, 2000, n),
        "slope": np.random.uniform(5, 45, n),
        "aspect": np.random.uniform(0, 360, n),
        "historical_landslide": [1 if i < 3 else 0 for i in range(n)],
        "rainfall_1h": np.random.uniform(0, 20, n),
        "rainfall_6h": np.random.uniform(0, 80, n),
        "rainfall_24h": np.random.uniform(0, 150, n),
        "rainfall_3d": np.random.uniform(0, 300, n),
        "rainfall_7d": np.random.uniform(0, 500, n),
        "forecast_rainfall": np.random.uniform(0, 50, n),
        "moisture_indicator": np.random.uniform(0, 1, n),
    })


class TestFeatureEngineer:

    def test_all_engineered_columns_added(self):
        df = make_full_df()
        fe = FeatureEngineer()
        result = fe.transform(df)
        for col in fe.engineered_column_names():
            assert col in result.columns, f"Expected column '{col}' in output"

    def test_does_not_modify_input(self):
        df = make_full_df()
        original_cols = list(df.columns)
        fe = FeatureEngineer()
        fe.transform(df)
        assert list(df.columns) == original_cols  # input unchanged

    def test_reproducible(self):
        df = make_full_df()
        fe = FeatureEngineer()
        r1 = fe.transform(df)
        r2 = fe.transform(df)
        for col in fe.engineered_column_names():
            pd.testing.assert_series_equal(r1[col], r2[col])

    def test_rainfall_intensity_non_negative(self):
        df = make_full_df()
        result = FeatureEngineer().transform(df)
        assert (result["rainfall_intensity"] >= 0).all()

    def test_antecedent_rainfall_equals_3d_plus_7d(self):
        df = make_full_df()
        result = FeatureEngineer().transform(df)
        expected = df["rainfall_3d"] + df["rainfall_7d"]
        pd.testing.assert_series_equal(
            result["antecedent_rainfall"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_susceptibility_score_in_range(self):
        df = make_full_df()
        result = FeatureEngineer().transform(df)
        score = result["static_susceptibility_score"]
        assert (score >= 0).all(), "Susceptibility score must be non-negative"
        assert (score <= 1.1).all(), "Susceptibility score should be near [0,1]"

    def test_rain_x_slope_correct(self):
        df = make_full_df()
        result = FeatureEngineer().transform(df)
        expected = df["rainfall_24h"] * df["slope"]
        pd.testing.assert_series_equal(
            result["rain_x_slope"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_missing_rainfall_columns_produce_nan(self):
        """Missing inputs must produce NaN outputs, not zeros."""
        df = make_full_df().drop(columns=["rainfall_1h", "rainfall_6h"])
        result = FeatureEngineer().transform(df)
        assert result["rainfall_intensity"].isna().all(), \
            "rainfall_intensity should be NaN when inputs are missing"

    def test_missing_slope_column_produces_nan(self):
        df = make_full_df().drop(columns=["slope"])
        result = FeatureEngineer().transform(df)
        assert result["rain_x_slope"].isna().all()

    def test_engineer_features_convenience(self):
        df = make_full_df()
        result = engineer_features(df)
        assert "rainfall_intensity" in result.columns
        assert "antecedent_rainfall" in result.columns

    def test_engineered_column_names_returns_list(self):
        fe = FeatureEngineer()
        names = fe.engineered_column_names()
        assert isinstance(names, list)
        assert len(names) > 0

    def test_single_row_works(self):
        """Single-row inference-like input must not crash."""
        df = make_full_df(n=1)
        result = FeatureEngineer().transform(df)
        assert len(result) == 1
        for col in FeatureEngineer().engineered_column_names():
            assert col in result.columns
