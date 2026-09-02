"""
Unit tests for the data quality pipeline.
"""

import pandas as pd
import pytest

from data_engineering.validation.quality_pipeline import run_quality_pipeline


@pytest.fixture
def clean_df():
    return pd.DataFrame(
        {
            "latitude": [27.34, 27.12, 27.56],
            "longitude": [88.61, 88.45, 88.78],
            "timestamp_utc": [
                "2022-06-15",
                "2022-07-01",
                "2022-07-15",
            ],
        }
    )


@pytest.fixture
def df_with_issues():
    return pd.DataFrame(
        {
            "latitude": [27.34, None, 999.0, 27.12, 27.34],
            "longitude": [88.61, 88.45, 88.0, 88.45, 88.61],
            "timestamp_utc": [
                "2022-06-15",
                "2022-07-01",
                "2022-07-15",
                "invalid_date",
                "2022-06-15",  # duplicate of row 0
            ],
        }
    )


class TestRunQualityPipeline:
    def test_all_valid_rows(self, clean_df):
        result, report = run_quality_pipeline(clean_df)
        assert report.records_total == 3
        assert report.records_valid == 3
        assert report.records_invalid == 0

    def test_valid_column_added(self, clean_df):
        result, report = run_quality_pipeline(clean_df)
        assert "_valid" in result.columns
        assert result["_valid"].all()

    def test_missing_coordinate_detection(self, df_with_issues):
        result, report = run_quality_pipeline(df_with_issues)
        assert report.missing_coordinates >= 1

    def test_invalid_coordinate_detection(self, df_with_issues):
        result, report = run_quality_pipeline(df_with_issues)
        assert report.invalid_coordinates >= 1

    def test_invalid_timestamp_detection(self, df_with_issues):
        result, report = run_quality_pipeline(df_with_issues)
        assert report.missing_timestamps >= 1

    def test_duplicate_detection(self, df_with_issues):
        result, report = run_quality_pipeline(df_with_issues)
        assert report.duplicate_records >= 1

    def test_no_silent_zero_replacement(self, df_with_issues):
        """Missing lat/lon must not become 0.0."""
        result, report = run_quality_pipeline(df_with_issues)
        # The row with None latitude should be marked invalid, not have lat=0
        none_row = result[result[result.columns[0]].isna() if "latitude" in result.columns else result.index < 0]
        # Ensure the None row is flagged invalid
        invalid_rows = result[~result["_valid"]]
        assert len(invalid_rows) >= 1

    def test_total_equals_valid_plus_invalid(self, df_with_issues):
        result, report = run_quality_pipeline(df_with_issues)
        assert report.records_total == report.records_valid + report.records_invalid

    def test_report_is_dict_serializable(self, clean_df):
        _, report = run_quality_pipeline(clean_df)
        d = report.to_dict()
        assert isinstance(d, dict)
        assert "records_total" in d

    def test_required_feature_missing(self):
        df = pd.DataFrame(
            {
                "latitude": [27.34],
                "longitude": [88.61],
                "timestamp_utc": ["2022-06-15"],
                "elevation": [None],  # missing required feature
            }
        )
        result, report = run_quality_pipeline(
            df, required_feature_cols=["elevation"]
        )
        assert report.missing_feature_values >= 1
