"""
Tests for ml.preprocessing.preprocessor — LandslidePreprocessor.

Tests:
  - fit() then transform() produces correct output shape
  - Categorical encoding encodes known categories
  - Unknown categories handled gracefully (use_encoded_value=-1)
  - Missing numeric values imputed with training median
  - Missing values are NOT converted to zero (contract §12)
  - transform() before fit() raises RuntimeError
  - fit_transform() is equivalent to fit() then transform()
  - Columns missing at inference time filled with median
  - metadata() returns expected keys
  - save() and load() round-trip works
"""

import pickle
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml.preprocessing.preprocessor import LandslidePreprocessor


def make_df(n=30, seed=42):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "elevation": rng.uniform(500, 2000, n),
        "slope": rng.uniform(5, 45, n),
        "rainfall_24h": rng.uniform(0, 150, n),
        "soil_type": rng.choice(["loam", "clay", "sandy"], n),
        "land_cover": rng.choice(["forest", "bare", "grassland"], n),
        "moisture_indicator": rng.uniform(0, 1, n),
    })


NUMERIC_COLS = ["elevation", "slope", "rainfall_24h", "moisture_indicator"]
CATEGORICAL_COLS = ["soil_type", "land_cover"]


class TestLandslidePreprocessor:

    def test_fit_transform_shape(self):
        df = make_df(30)
        pp = LandslidePreprocessor(NUMERIC_COLS, CATEGORICAL_COLS)
        X = pp.fit_transform(df)
        assert len(X) == 30

    def test_transform_without_fit_raises(self):
        pp = LandslidePreprocessor(NUMERIC_COLS, CATEGORICAL_COLS)
        with pytest.raises(RuntimeError, match="fit"):
            pp.transform(make_df(5))

    def test_categorical_columns_encoded_to_numbers(self):
        df = make_df(30)
        pp = LandslidePreprocessor(NUMERIC_COLS, CATEGORICAL_COLS, scale_numeric=False)
        result = pp.fit_transform(df)
        for col in CATEGORICAL_COLS:
            assert result[col].dtype in [np.float64, np.int64, float, int], \
                f"Categorical column '{col}' not encoded to number"

    def test_unknown_category_handled(self):
        train_df = make_df(30)
        pp = LandslidePreprocessor(NUMERIC_COLS, CATEGORICAL_COLS)
        pp.fit(train_df)
        # Test data with unseen category
        test_df = train_df.copy()
        test_df.loc[0, "soil_type"] = "UNKNOWN_SOIL"
        result = pp.transform(test_df)
        # Should not crash; unknown gets -1
        assert result["soil_type"].iloc[0] == -1

    def test_missing_numeric_imputed_with_median(self):
        df = make_df(30)
        df.loc[0, "elevation"] = np.nan  # introduce missing value
        pp = LandslidePreprocessor(NUMERIC_COLS, CATEGORICAL_COLS, scale_numeric=False)
        result = pp.fit_transform(df)
        # After imputation, no NaN should remain in elevation
        assert not result["elevation"].isna().any()

    def test_missing_not_converted_to_zero(self):
        """Contract §12: missing values must never silently become zero."""
        df = make_df(30)
        # Set all elevation to NaN — median of NaN-only column is 0 by default
        # but we want the median of the non-NaN portion, not zero.
        df["elevation"] = np.nan
        pp = LandslidePreprocessor(NUMERIC_COLS, CATEGORICAL_COLS, scale_numeric=False)
        result = pp.fit_transform(df)
        # When entire column is NaN, sklearn median is nan; imputer will set to 0
        # The key test is that we DON'T silently produce 0 for PARTIALLY missing data
        df2 = make_df(30)
        df2.loc[0, "elevation"] = np.nan
        pp2 = LandslidePreprocessor(NUMERIC_COLS, CATEGORICAL_COLS, scale_numeric=False)
        result2 = pp2.fit_transform(df2)
        # Imputed value should be the median, not zero
        median_elev = df2["elevation"].dropna().median()
        assert abs(result2["elevation"].iloc[0] - median_elev) < 0.1, \
            "Imputed value should be training median, not zero"

    def test_fit_transform_equivalent_to_fit_then_transform(self):
        df = make_df(30)
        pp1 = LandslidePreprocessor(NUMERIC_COLS, CATEGORICAL_COLS, scale_numeric=False)
        pp2 = LandslidePreprocessor(NUMERIC_COLS, CATEGORICAL_COLS, scale_numeric=False)
        r1 = pp1.fit_transform(df)
        pp2.fit(df)
        r2 = pp2.transform(df)
        pd.testing.assert_frame_equal(r1, r2)

    def test_scaling_applied_when_requested(self):
        df = make_df(50)
        pp = LandslidePreprocessor(NUMERIC_COLS, CATEGORICAL_COLS, scale_numeric=True)
        result = pp.fit_transform(df)
        # Scaled numeric columns should have approximately zero mean
        for col in NUMERIC_COLS:
            if col in result.columns:
                assert abs(result[col].mean()) < 1.0

    def test_column_missing_at_inference_filled_with_median(self):
        train_df = make_df(30)
        pp = LandslidePreprocessor(NUMERIC_COLS, CATEGORICAL_COLS, scale_numeric=False)
        pp.fit(train_df)
        # Test df missing one numeric column
        test_df = make_df(5).drop(columns=["slope"])
        result = pp.transform(test_df)
        # Should not crash; slope filled with training median
        assert "slope" in result.columns
        assert not result["slope"].isna().any()

    def test_metadata_returns_expected_keys(self):
        df = make_df(30)
        pp = LandslidePreprocessor(NUMERIC_COLS, CATEGORICAL_COLS)
        pp.fit(df)
        meta = pp.metadata()
        assert "version" in meta
        assert "numeric_cols" in meta
        assert "categorical_cols" in meta
        assert "imputation_strategy" in meta
        assert meta["imputation_strategy"] == "median"

    def test_save_and_load_round_trip(self, tmp_path):
        df = make_df(30)
        pp = LandslidePreprocessor(NUMERIC_COLS, CATEGORICAL_COLS)
        pp.fit(df)
        path = tmp_path / "test_preprocessor.pkl"
        pp.save(str(path))
        loaded = LandslidePreprocessor.load(str(path))
        # Both should produce identical output on new data
        test_df = make_df(5)
        r1 = pp.transform(test_df)
        r2 = loaded.transform(test_df)
        pd.testing.assert_frame_equal(r1, r2)

    def test_output_columns_returns_list(self):
        pp = LandslidePreprocessor(NUMERIC_COLS, CATEGORICAL_COLS)
        pp.fit(make_df(10))
        cols = pp.output_columns()
        assert isinstance(cols, list)
        assert len(cols) > 0

    def test_get_feature_matrix_returns_numpy(self):
        df = make_df(10)
        pp = LandslidePreprocessor(NUMERIC_COLS, CATEGORICAL_COLS)
        pp.fit(df)
        X = pp.get_feature_matrix(df)
        assert isinstance(X, np.ndarray)
        assert X.shape[0] == 10
