"""
Unit tests for duplicate detection.
"""

import pandas as pd
import pytest
from data_engineering.validation.duplicate_detector import detect_duplicates, count_duplicates


@pytest.fixture
def base_df():
    return pd.DataFrame(
        {
            "latitude": [27.3412, 27.1234, 27.3412, 27.5678],
            "longitude": [88.6123, 88.4560, 88.6123, 88.7890],
            "timestamp_utc": [
                "2022-06-15T00:00:00Z",
                "2022-07-01T00:00:00Z",
                "2022-06-15T00:00:00Z",  # duplicate of row 0
                "2022-07-15T00:00:00Z",
            ],
        }
    )


class TestDetectDuplicates:
    def test_identifies_duplicates(self, base_df):
        result = detect_duplicates(base_df)
        assert result["is_duplicate"].sum() == 1
        assert bool(result.loc[2, "is_duplicate"]) is True

    def test_first_occurrence_not_marked(self, base_df):
        result = detect_duplicates(base_df)
        assert bool(result.loc[0, "is_duplicate"]) is False

    def test_no_duplicates(self):
        df = pd.DataFrame(
            {
                "latitude": [27.1, 27.2],
                "longitude": [88.1, 88.2],
                "timestamp_utc": ["2022-01-01T00:00:00Z", "2022-01-02T00:00:00Z"],
            }
        )
        result = detect_duplicates(df)
        assert result["is_duplicate"].sum() == 0

    def test_does_not_drop_records(self, base_df):
        """Duplicates must be flagged, not dropped."""
        result = detect_duplicates(base_df)
        assert len(result) == len(base_df)

    def test_different_timestamps_not_duplicate(self):
        df = pd.DataFrame(
            {
                "latitude": [27.34, 27.34],
                "longitude": [88.61, 88.61],
                "timestamp_utc": ["2022-06-15T00:00:00Z", "2023-06-15T00:00:00Z"],
            }
        )
        result = detect_duplicates(df)
        assert result["is_duplicate"].sum() == 0

    def test_coord_precision(self):
        """Points within precision threshold are treated as duplicates."""
        df = pd.DataFrame(
            {
                # Both round to 27.3412 at 4 d.p. → treated as same location
                "latitude": [27.34124, 27.34125],
                "longitude": [88.61230, 88.61230],
                "timestamp_utc": ["2022-06-15T00:00:00Z", "2022-06-15T00:00:00Z"],
            }
        )
        # At 4 d.p. 27.34124 → 27.3412, 27.34125 → 27.3413 (banker's rounding)
        # Use values that definitely round identically
        df2 = pd.DataFrame(
            {
                "latitude": [27.34121, 27.34122],  # both round to 27.3412
                "longitude": [88.61231, 88.61231],
                "timestamp_utc": ["2022-06-15T00:00:00Z", "2022-06-15T00:00:00Z"],
            }
        )
        result = detect_duplicates(df2, coord_precision=4)
        assert result["is_duplicate"].sum() == 1

    def test_count_duplicates(self, base_df):
        assert count_duplicates(base_df) == 1
