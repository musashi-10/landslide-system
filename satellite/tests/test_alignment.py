"""
satellite/tests/test_alignment.py
Tests for satellite/spatial/alignment.py
"""

import numpy as np
import pytest

from satellite.spatial.alignment import (
    make_location_id,
    raster_to_points,
    assign_location_id,
)
from satellite.tests.conftest import ACQUISITION_TIME, SOURCE_CRS


class TestMakeLocationId:
    def test_format(self):
        """location_id must follow LOC_LAT{:.4f}_LON{:.4f} format."""
        lid = make_location_id(27.123, 88.456)
        assert lid == "LOC_LAT27.1230_LON88.4560"

    def test_negative_lat(self):
        lid = make_location_id(-1.5, 37.25)
        assert lid == "LOC_LAT-1.5000_LON37.2500"

    def test_deterministic(self):
        """Same lat/lon → same location_id every time."""
        a = make_location_id(27.0, 88.0)
        b = make_location_id(27.0, 88.0)
        assert a == b

    def test_different_coords_different_ids(self):
        a = make_location_id(27.0, 88.0)
        b = make_location_id(27.0001, 88.0)
        assert a != b

    def test_returns_string(self):
        assert isinstance(make_location_id(0.0, 0.0), str)


class TestRasterToPoints:
    """Tests for raster_to_points using the synthetic fixtures."""

    def _make_feature_arrays(self, shape=(10, 10)):
        rng = np.random.default_rng(seed=7)
        ndvi = np.ma.array(
            rng.uniform(-0.2, 0.8, shape).astype(np.float32)
        )
        bsi = np.ma.array(
            rng.uniform(-0.3, 0.3, shape).astype(np.float32)
        )
        return {"ndvi": ndvi, "bare_surface_index": bsi}

    def test_records_have_location_id(self, affine_transform):
        feats = self._make_feature_arrays()
        records = raster_to_points(feats, affine_transform, SOURCE_CRS, ACQUISITION_TIME)
        assert len(records) > 0
        for rec in records:
            assert "location_id" in rec
            assert rec["location_id"].startswith("LOC_LAT")

    def test_records_have_lat_lon(self, affine_transform):
        feats = self._make_feature_arrays()
        records = raster_to_points(feats, affine_transform, SOURCE_CRS, ACQUISITION_TIME)
        for rec in records:
            assert "latitude" in rec
            assert "longitude" in rec
            assert -90 <= rec["latitude"] <= 90
            assert -180 <= rec["longitude"] <= 180

    def test_output_crs_is_epsg4326(self, affine_transform):
        feats = self._make_feature_arrays()
        records = raster_to_points(feats, affine_transform, SOURCE_CRS, ACQUISITION_TIME)
        for rec in records:
            assert rec["output_crs"] == "EPSG:4326"

    def test_source_crs_recorded(self, affine_transform):
        feats = self._make_feature_arrays()
        records = raster_to_points(feats, affine_transform, SOURCE_CRS, ACQUISITION_TIME)
        for rec in records:
            assert rec["source_crs"] == SOURCE_CRS

    def test_acquisition_time_present(self, affine_transform):
        feats = self._make_feature_arrays()
        records = raster_to_points(feats, affine_transform, SOURCE_CRS, ACQUISITION_TIME)
        for rec in records:
            assert rec["acquisition_time"] == ACQUISITION_TIME

    def test_masked_pixels_excluded(self, affine_transform):
        """Masked pixels must not appear in the output records."""
        ndvi_mask = np.zeros((10, 10), dtype=bool)
        ndvi_mask[:] = True  # mask all
        ndvi = np.ma.array(np.ones((10, 10), dtype=np.float32), mask=ndvi_mask)
        feats = {"ndvi": ndvi}
        records = raster_to_points(feats, affine_transform, SOURCE_CRS, ACQUISITION_TIME)
        assert len(records) == 0

    def test_all_valid_pixels_included(self, affine_transform):
        """All 100 valid pixels of a 10×10 all-valid array → 100 records."""
        ndvi = np.ma.array(np.full((10, 10), 0.5, dtype=np.float32))
        feats = {"ndvi": ndvi}
        records = raster_to_points(feats, affine_transform, SOURCE_CRS, ACQUISITION_TIME)
        assert len(records) == 100

    def test_inconsistent_shapes_raise(self, affine_transform):
        """Feature arrays with different shapes must raise ValueError."""
        ndvi = np.ma.zeros((10, 10), dtype=np.float32)
        bsi = np.ma.zeros((5, 5), dtype=np.float32)
        with pytest.raises(ValueError, match="shape"):
            raster_to_points({"ndvi": ndvi, "bsi": bsi}, affine_transform, SOURCE_CRS, ACQUISITION_TIME)

    def test_geometry_is_wkt_point(self, affine_transform):
        """geometry field must be a WKT POINT string."""
        ndvi = np.ma.array(np.full((2, 2), 0.5, dtype=np.float32))
        records = raster_to_points({"ndvi": ndvi}, affine_transform, SOURCE_CRS, ACQUISITION_TIME)
        for rec in records:
            assert rec["geometry"].startswith("POINT")

    def test_empty_feature_arrays(self, affine_transform):
        """Empty feature dict → empty records list."""
        records = raster_to_points({}, affine_transform, SOURCE_CRS, ACQUISITION_TIME)
        assert records == []

    def test_provenance_fields_present(self, affine_transform):
        ndvi = np.ma.array(np.full((2, 2), 0.4, dtype=np.float32))
        records = raster_to_points(
            {"ndvi": ndvi}, affine_transform, SOURCE_CRS, ACQUISITION_TIME,
            source="Sentinel-2", spatial_resolution_m=10, processing_version="v1",
        )
        for rec in records:
            assert rec["source"] == "Sentinel-2"
            assert rec["spatial_resolution_m"] == 10
            assert rec["processing_version"] == "v1"


class TestAssignLocationId:
    def test_missing_location_id_generated(self):
        """Records without location_id should get one assigned."""
        records = [{"latitude": 27.123, "longitude": 88.456}]
        result = assign_location_id(records)
        assert "location_id" in result[0]
        assert result[0]["location_id"] == "LOC_LAT27.1230_LON88.4560"

    def test_existing_location_id_preserved(self):
        """Records that already have location_id should not be overwritten."""
        records = [{"location_id": "CUSTOM_ID", "latitude": 27.0, "longitude": 88.0}]
        result = assign_location_id(records)
        assert result[0]["location_id"] == "CUSTOM_ID"

    def test_none_location_id_generated(self):
        """location_id=None should be replaced."""
        records = [{"location_id": None, "latitude": 27.0, "longitude": 88.0}]
        result = assign_location_id(records)
        assert result[0]["location_id"] is not None


class TestLocationIdJoinCompatibility:
    """Verify that location_id generated from same coordinates is consistent."""

    def test_same_coord_same_id(self, affine_transform):
        """Two calls with same coordinates must produce the same location_id."""
        ndvi = np.ma.array(np.full((1, 1), 0.5, dtype=np.float32))
        r1 = raster_to_points({"ndvi": ndvi}, affine_transform, SOURCE_CRS, ACQUISITION_TIME)
        r2 = raster_to_points({"ndvi": ndvi}, affine_transform, SOURCE_CRS, ACQUISITION_TIME)
        assert r1[0]["location_id"] == r2[0]["location_id"]

    def test_location_id_format_is_joinable(self, affine_transform):
        """location_id should contain no spaces — safe as a join key."""
        ndvi = np.ma.array(np.full((3, 3), 0.5, dtype=np.float32))
        records = raster_to_points({"ndvi": ndvi}, affine_transform, SOURCE_CRS, ACQUISITION_TIME)
        for rec in records:
            assert " " not in rec["location_id"]
