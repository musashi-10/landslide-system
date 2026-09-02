"""
Integration test: raw CSV fixture → ingestion → preprocessing → standardised output.

This test exercises the full pipeline from raw synthetic data to the
standardised Parquet-ready DataFrame that Engineer 5 (ML) and Engineer 2
(GIS) would consume.
"""

from pathlib import Path

import pandas as pd
import pytest

from data_engineering.pipelines import build_dataset
from data_engineering.spatial.grid import GridConfig

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SAMPLE_CSV = FIXTURES_DIR / "sample_landslides.csv"
SAMPLE_GEOJSON = FIXTURES_DIR / "sample_landslides.geojson"


class TestFullPipelineCSV:
    """
    End-to-end test: CSV fixture through the complete pipeline.
    """

    def test_pipeline_returns_dataframe(self):
        df, report, provenance = build_dataset(
            source_path=SAMPLE_CSV,
            column_map={"latitude": "latitude", "longitude": "longitude", "timestamp": "date"},
            source_name="sample_csv_test",
        )
        assert isinstance(df, pd.DataFrame)

    def test_required_columns_present(self):
        """Output must contain all contract-required columns."""
        df, _, _ = build_dataset(
            source_path=SAMPLE_CSV,
            column_map={"latitude": "latitude", "longitude": "longitude", "timestamp": "date"},
        )
        required = [
            "location_id",
            "latitude",
            "longitude",
            "geometry",
            "timestamp_utc",
            "historical_landslide",
            "source",
        ]
        for col in required:
            assert col in df.columns, f"Missing required column: {col}"

    def test_invalid_records_excluded(self):
        """Records with bad coords or timestamps must not appear in output."""
        df, report, _ = build_dataset(
            source_path=SAMPLE_CSV,
            column_map={"latitude": "latitude", "longitude": "longitude", "timestamp": "date"},
        )
        # Fixture has 20 rows; several are invalid
        assert report.records_invalid > 0
        # Invalid records are excluded from output df
        assert len(df) < 20

    def test_timestamps_are_utc_iso(self):
        df, _, _ = build_dataset(
            source_path=SAMPLE_CSV,
            column_map={"latitude": "latitude", "longitude": "longitude", "timestamp": "date"},
        )
        for ts in df["timestamp_utc"]:
            assert ts.endswith("Z"), f"Timestamp not UTC: {ts}"

    def test_location_id_format(self):
        df, _, _ = build_dataset(
            source_path=SAMPLE_CSV,
            column_map={"latitude": "latitude", "longitude": "longitude", "timestamp": "date"},
        )
        for loc_id in df["location_id"]:
            assert loc_id.startswith("LOC_"), f"Bad location_id: {loc_id}"

    def test_historical_landslide_is_one(self):
        df, _, _ = build_dataset(
            source_path=SAMPLE_CSV,
            column_map={"latitude": "latitude", "longitude": "longitude", "timestamp": "date"},
        )
        assert (df["historical_landslide"] == 1).all()

    def test_duplicates_flagged_not_dropped(self):
        df, report, _ = build_dataset(
            source_path=SAMPLE_CSV,
            column_map={"latitude": "latitude", "longitude": "longitude", "timestamp": "date"},
        )
        assert "is_duplicate" in df.columns
        # Fixture has intentional duplicates
        assert report.duplicate_records > 0

    def test_provenance_source_matches(self):
        _, _, provenance = build_dataset(
            source_path=SAMPLE_CSV,
            column_map={"latitude": "latitude", "longitude": "longitude", "timestamp": "date"},
            source_name="my_source",
        )
        assert provenance.source == "my_source"

    def test_with_grid_config(self):
        grid = GridConfig(
            min_lat=27.0, max_lat=28.0,
            min_lon=88.0, max_lon=89.0,
            resolution_deg=0.5,
        )
        df, _, _ = build_dataset(
            source_path=SAMPLE_CSV,
            column_map={"latitude": "latitude", "longitude": "longitude", "timestamp": "date"},
            grid_config=grid,
        )
        assert "grid_id" in df.columns


class TestFullPipelineGeoJSON:
    def test_geojson_pipeline(self):
        df, report, provenance = build_dataset(
            source_path=SAMPLE_GEOJSON,
            column_map={"latitude": "latitude", "longitude": "longitude", "timestamp": "date"},
        )
        assert isinstance(df, pd.DataFrame)
        # GeoJSON fixture has 3 valid features
        assert len(df) >= 1

    def test_geojson_required_columns(self):
        df, _, _ = build_dataset(
            source_path=SAMPLE_GEOJSON,
            column_map={"latitude": "latitude", "longitude": "longitude", "timestamp": "date"},
        )
        required = ["location_id", "latitude", "longitude", "timestamp_utc"]
        for col in required:
            assert col in df.columns


class TestValidationReport:
    def test_report_counts_consistent(self):
        _, report, _ = build_dataset(
            source_path=SAMPLE_CSV,
            column_map={"latitude": "latitude", "longitude": "longitude", "timestamp": "date"},
        )
        d = report.to_dict()
        assert d["records_total"] == d["records_valid"] + d["records_invalid"]

    def test_report_missing_coordinates_counted(self):
        _, report, _ = build_dataset(
            source_path=SAMPLE_CSV,
            column_map={"latitude": "latitude", "longitude": "longitude", "timestamp": "date"},
        )
        # Fixture rows LS016, LS017 have missing lat/lon
        assert report.missing_coordinates >= 2

    def test_report_invalid_coordinates_counted(self):
        _, report, _ = build_dataset(
            source_path=SAMPLE_CSV,
            column_map={"latitude": "latitude", "longitude": "longitude", "timestamp": "date"},
        )
        # Fixture rows LS018 (lat=999), LS019 (lon=-999)
        assert report.invalid_coordinates >= 2
