"""
environment/tests/test_spatial_mapper.py

Tests for spatial mapping utilities.
"""

from __future__ import annotations

import pytest

from environment.preprocessing.spatial_mapper import (
    ProjectLocation,
    SpatialMapper,
    generate_location_id,
    haversine_km,
)


class TestGenerateLocationId:
    def test_format_positive_coords(self):
        loc_id = generate_location_id(27.123, 88.456)
        # Engineer 1 uses {:+08.4f} which gives +88.4560 (8 chars total inc. sign)
        assert loc_id == "LOC_+27.1230_+88.4560"

    def test_format_negative_coords(self):
        loc_id = generate_location_id(-34.5678, -58.1234)
        assert loc_id.startswith("LOC_-")
        assert "-34.5678" in loc_id

    def test_zero_coords(self):
        loc_id = generate_location_id(0.0, 0.0)
        assert "LOC_" in loc_id

    def test_deterministic(self):
        id1 = generate_location_id(27.123, 88.456)
        id2 = generate_location_id(27.123, 88.456)
        assert id1 == id2

    def test_engineer1_compatible(self):
        """Must match Engineer 1's generate_location_id format."""
        loc_id = generate_location_id(27.123, 88.456)
        # Engineer 1 format: LOC_{lat:+.4f}_{lon:+08.4f}
        # {:+08.4f} for lon=88.456 -> '+88.4560' (8 chars including sign)
        assert loc_id == "LOC_+27.1230_+88.4560"


class TestHaversineKm:
    def test_same_point_is_zero(self):
        dist = haversine_km(27.0, 88.0, 27.0, 88.0)
        assert dist == pytest.approx(0.0, abs=0.001)

    def test_known_distance(self):
        # London to Paris ≈ 342 km
        dist = haversine_km(51.5074, -0.1278, 48.8566, 2.3522)
        assert dist == pytest.approx(342.0, rel=0.05)  # 5% tolerance

    def test_positive_result(self):
        dist = haversine_km(27.0, 88.0, 28.0, 89.0)
        assert dist > 0


class TestSpatialMapper:
    def test_pass_through_mode_no_locations(self):
        mapper = SpatialMapper()
        loc_id, lat, lon = mapper.map(27.123, 88.456)
        assert loc_id == "LOC_+27.1230_+88.4560"
        assert lat == pytest.approx(27.123)
        assert lon == pytest.approx(88.456)

    def test_exact_match(self):
        locs = [
            ProjectLocation("LOC_A", 27.123, 88.456),
            ProjectLocation("LOC_B", 28.000, 89.000),
        ]
        mapper = SpatialMapper(project_locations=locs, max_distance_km=5.0)
        loc_id, lat, lon = mapper.map(27.123, 88.456)
        assert loc_id == "LOC_A"

    def test_nearest_neighbour(self):
        locs = [
            ProjectLocation("LOC_A", 27.000, 88.000),
            ProjectLocation("LOC_B", 28.000, 89.000),
        ]
        mapper = SpatialMapper(project_locations=locs, max_distance_km=200.0)
        # 27.01, 88.01 is closer to LOC_A
        loc_id, _, _ = mapper.map(27.01, 88.01)
        assert loc_id == "LOC_A"

    def test_falls_back_when_too_far(self):
        """If nearest project location exceeds max_distance, use direct location_id."""
        locs = [
            ProjectLocation("LOC_A", 27.000, 88.000),
        ]
        mapper = SpatialMapper(project_locations=locs, max_distance_km=1.0)
        # 30.0, 90.0 is ~400 km from LOC_A
        loc_id, lat, lon = mapper.map(30.0, 90.0)
        assert loc_id == generate_location_id(30.0, 90.0)
        assert lat == pytest.approx(30.0)
        assert lon == pytest.approx(90.0)

    def test_empty_project_locations(self):
        mapper = SpatialMapper(project_locations=[], max_distance_km=5.0)
        loc_id, _, _ = mapper.map(27.123, 88.456)
        assert loc_id == "LOC_+27.1230_+88.4560"
