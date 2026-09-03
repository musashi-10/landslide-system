"""
Tests for ml.training.validation_strategy — SpatioTemporalSplitter.

Tests:
  - Temporal split returns train/val/test splits summing to total length
  - Temporal ordering is maintained (train < val < test timestamps)
"""

import numpy as np
import pandas as pd
import pytest

from ml.training.validation_strategy import SpatioTemporalSplitter, ValidationStrategy


def make_temporal_df(n_samples=100):
    rows = []
    # Create timestamps sequentially to test temporal split
    for i in range(n_samples):
        rows.append({
            "location_id": f"LOC_{i%5:03d}",
            "timestamp_utc": f"2026-09-{(i//5)+1:02d}T00:00:00Z",
            "latitude": 27.0 + (i % 5) * 0.1,
        })
    df = pd.DataFrame(rows)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"])
    return df


class TestSpatioTemporalSplitter:

    def test_temporal_split_sizes(self):
        df = make_temporal_df(100)
        splitter = SpatioTemporalSplitter(
            strategy=ValidationStrategy.TEMPORAL,
            temporal_test_fraction=0.20,
            temporal_val_fraction=0.10,
        )
        
        result = splitter.split(df)
        
        # Test 20%, Val 10%, Train 70%
        assert result.n_test == 20
        assert result.n_val == 10
        assert result.n_train == 70
        assert result.n_train + result.n_val + result.n_test == 100

    def test_temporal_ordering_preserved(self):
        df = make_temporal_df(100)
        # Ensure it's not sorted initially to test the sorting logic
        df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        
        splitter = SpatioTemporalSplitter(
            strategy=ValidationStrategy.TEMPORAL
        )
        result = splitter.split(df)
        
        # Extract max/min timestamps for each split
        train_max = df.iloc[result.train_idx]["timestamp_utc"].max()
        val_min = df.iloc[result.val_idx]["timestamp_utc"].min()
        val_max = df.iloc[result.val_idx]["timestamp_utc"].max()
        test_min = df.iloc[result.test_idx]["timestamp_utc"].min()
        
        assert train_max <= val_min
        assert val_max <= test_min

    def test_split_result_summary(self):
        df = make_temporal_df(10)
        splitter = SpatioTemporalSplitter()
        result = splitter.split(df)
        
        summary = result.summary()
        assert "n_train" in summary
        assert "n_val" in summary
        assert "n_test" in summary
