"""
satellite/tests/test_schema.py
Tests for satellite/schemas/satellite_schema.py
"""

import pytest
from pydantic import ValidationError

from satellite.schemas.satellite_schema import SatelliteFeatureRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_valid_record(**overrides) -> dict:
    """Return a valid SatelliteFeatureRecord dict with optional overrides."""
    base = {
        "location_id": "LOC_LAT27.1230_LON88.4560",
        "latitude": 27.123,
        "longitude": 88.456,
        "acquisition_time": "2024-06-15T10:23:00Z",
        "source": "Sentinel-2",
        "spatial_resolution_m": 10,
        "processing_version": "v1",
        "output_crs": "EPSG:4326",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Valid record tests
# ---------------------------------------------------------------------------

class TestSatelliteFeatureRecordValid:
    def test_minimal_valid_record(self):
        """Record with only required fields should validate."""
        rec = SatelliteFeatureRecord(**make_valid_record())
        assert rec.location_id == "LOC_LAT27.1230_LON88.4560"

    def test_all_optional_fields_none(self):
        """Optional satellite features can all be None."""
        rec = SatelliteFeatureRecord(**make_valid_record())
        assert rec.ndvi is None
        assert rec.bare_surface_index is None
        assert rec.ndvi_change is None
        assert rec.land_cover is None

    def test_with_ndvi(self):
        rec = SatelliteFeatureRecord(**make_valid_record(ndvi=0.65))
        assert rec.ndvi == 0.65

    def test_with_bsi(self):
        rec = SatelliteFeatureRecord(**make_valid_record(bare_surface_index=0.12))
        assert rec.bare_surface_index == 0.12

    def test_with_ndvi_change(self):
        rec = SatelliteFeatureRecord(**make_valid_record(ndvi_change=-0.25))
        assert rec.ndvi_change == -0.25

    def test_with_all_land_cover_classes(self):
        """Each valid land_cover value should pass validation."""
        valid_classes = ["forest", "shrub_grass", "bare", "water", "unknown"]
        for lc in valid_classes:
            rec = SatelliteFeatureRecord(**make_valid_record(land_cover=lc))
            assert rec.land_cover == lc

    def test_geometry_optional(self):
        rec = SatelliteFeatureRecord(**make_valid_record(geometry="POINT (88.456 27.123)"))
        assert rec.geometry == "POINT (88.456 27.123)"

    def test_source_crs_optional(self):
        rec = SatelliteFeatureRecord(**make_valid_record(source_crs="EPSG:32645"))
        assert rec.source_crs == "EPSG:32645"

    def test_acquisition_time_iso_format(self):
        """acquisition_time should be stored as provided (ISO 8601)."""
        rec = SatelliteFeatureRecord(**make_valid_record(acquisition_time="2024-06-15T10:23:00Z"))
        assert "2024-06-15" in rec.acquisition_time

    def test_ndvi_boundary_minus_one(self):
        rec = SatelliteFeatureRecord(**make_valid_record(ndvi=-1.0))
        assert rec.ndvi == -1.0

    def test_ndvi_boundary_plus_one(self):
        rec = SatelliteFeatureRecord(**make_valid_record(ndvi=1.0))
        assert rec.ndvi == 1.0


# ---------------------------------------------------------------------------
# Invalid record tests
# ---------------------------------------------------------------------------

class TestSatelliteFeatureRecordInvalid:
    def test_ndvi_above_one_rejected(self):
        """NDVI > 1.0 should fail validation."""
        with pytest.raises(ValidationError, match="range"):
            SatelliteFeatureRecord(**make_valid_record(ndvi=1.01))

    def test_ndvi_below_minus_one_rejected(self):
        """NDVI < −1.0 should fail validation."""
        with pytest.raises(ValidationError, match="range"):
            SatelliteFeatureRecord(**make_valid_record(ndvi=-1.01))

    def test_bsi_out_of_range_rejected(self):
        with pytest.raises(ValidationError, match="range"):
            SatelliteFeatureRecord(**make_valid_record(bare_surface_index=1.5))

    def test_unknown_land_cover_rejected(self):
        with pytest.raises(ValidationError, match="land_cover"):
            SatelliteFeatureRecord(**make_valid_record(land_cover="jungle"))

    def test_wrong_output_crs_rejected(self):
        """output_crs != 'EPSG:4326' must be rejected."""
        with pytest.raises(ValidationError, match="EPSG:4326"):
            SatelliteFeatureRecord(**make_valid_record(output_crs="EPSG:32645"))

    def test_latitude_out_of_range_rejected(self):
        with pytest.raises(ValidationError, match="latitude"):
            SatelliteFeatureRecord(**make_valid_record(latitude=95.0))

    def test_longitude_out_of_range_rejected(self):
        with pytest.raises(ValidationError, match="longitude"):
            SatelliteFeatureRecord(**make_valid_record(longitude=200.0))

    def test_zero_resolution_rejected(self):
        with pytest.raises(ValidationError, match="positive"):
            SatelliteFeatureRecord(**make_valid_record(spatial_resolution_m=0))

    def test_missing_location_id_rejected(self):
        data = make_valid_record()
        del data["location_id"]
        with pytest.raises(ValidationError):
            SatelliteFeatureRecord(**data)

    def test_missing_acquisition_time_rejected(self):
        data = make_valid_record()
        del data["acquisition_time"]
        with pytest.raises(ValidationError):
            SatelliteFeatureRecord(**data)


# ---------------------------------------------------------------------------
# Data contract: missing values must never be zero
# ---------------------------------------------------------------------------

class TestMissingValueContract:
    def test_missing_ndvi_is_none_not_zero(self):
        """Per data contract Section 12: missing values must not silently be 0."""
        rec = SatelliteFeatureRecord(**make_valid_record())
        # ndvi not provided → must be None, not 0.0
        assert rec.ndvi is None
        assert rec.ndvi != 0.0

    def test_missing_bsi_is_none_not_zero(self):
        rec = SatelliteFeatureRecord(**make_valid_record())
        assert rec.bare_surface_index is None

    def test_missing_ndvi_change_is_none_not_zero(self):
        rec = SatelliteFeatureRecord(**make_valid_record())
        assert rec.ndvi_change is None
