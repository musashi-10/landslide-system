"""
Tests for ml.training.target — target definition and leakage documentation.

Tests:
  - TargetBuilder.build() adds target column
  - Correct label from historical_landslide
  - Missing historical_landslide column raises ValueError
  - describe() returns required keys
  - TargetConfig defaults are sensible
"""

import pandas as pd
import numpy as np
import pytest

from ml.training.target import TargetBuilder, TargetConfig


def make_df(n=20):
    return pd.DataFrame({
        "location_id": [f"LOC_{i:03d}" for i in range(n)],
        "timestamp_utc": ["2026-09-01T00:00:00Z"] * n,
        "historical_landslide": [1 if i < 3 else 0 for i in range(n)],
        "elevation": np.random.uniform(500, 2000, n),
        "slope": np.random.uniform(5, 45, n),
        "rainfall_24h": np.random.uniform(0, 150, n),
    })


class TestTargetBuilder:

    def test_target_column_added(self):
        df = make_df(20)
        builder = TargetBuilder()
        result = builder.build(df)
        assert builder.config.target_col in result.columns

    def test_positive_labels_correct(self):
        df = make_df(20)  # 3 positives
        builder = TargetBuilder()
        result = builder.build(df)
        n_pos = int((result[builder.config.target_col] == 1).sum())
        assert n_pos == 3

    def test_negative_labels_correct(self):
        df = make_df(20)  # 17 negatives
        builder = TargetBuilder()
        result = builder.build(df)
        n_neg = int((result[builder.config.target_col] == 0).sum())
        assert n_neg == 17

    def test_missing_historical_landslide_raises(self):
        df = make_df(20).drop(columns=["historical_landslide"])
        builder = TargetBuilder()
        with pytest.raises(ValueError, match="historical_landslide"):
            builder.build(df)

    def test_does_not_modify_input(self):
        df = make_df(20)
        original_cols = list(df.columns)
        builder = TargetBuilder()
        builder.build(df)
        assert list(df.columns) == original_cols

    def test_describe_returns_required_keys(self):
        builder = TargetBuilder()
        desc = builder.describe()
        assert "positive_definition" in desc
        assert "negative_definition" in desc
        assert "temporal_window" in desc
        assert "leakage_guards" in desc
        assert "assumptions" in desc

    def test_describe_leakage_guards_is_list(self):
        builder = TargetBuilder()
        desc = builder.describe()
        assert isinstance(desc["leakage_guards"], list)
        assert len(desc["leakage_guards"]) >= 2

    def test_config_defaults_sensible(self):
        cfg = TargetConfig()
        assert cfg.event_buffer_days > 0
        assert cfg.event_buffer_km > 0
        assert cfg.event_location_match_km > 0
        assert cfg.target_col == "landslide_occurred"

    def test_custom_target_col_name(self):
        df = make_df(10)
        cfg = TargetConfig(target_col="my_target")
        builder = TargetBuilder(config=cfg)
        result = builder.build(df)
        assert "my_target" in result.columns
