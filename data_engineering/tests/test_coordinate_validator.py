"""
Unit tests for coordinate validation.
"""

import pytest
from data_engineering.validation.coordinate_validator import (
    validate_coordinates,
    is_valid_coordinate_pair,
    CoordinateError,
)


class TestValidateCoordinates:
    def test_valid_pair(self):
        lat, lon = validate_coordinates(27.34, 88.61)
        assert lat == pytest.approx(27.34)
        assert lon == pytest.approx(88.61)

    def test_valid_negative_coordinates(self):
        lat, lon = validate_coordinates(-34.56, -58.12)
        assert lat == pytest.approx(-34.56)

    def test_valid_boundary_values(self):
        assert validate_coordinates(90.0, 180.0) == (90.0, 180.0)
        assert validate_coordinates(-90.0, -180.0) == (-90.0, -180.0)

    def test_missing_latitude(self):
        with pytest.raises(CoordinateError, match="Missing"):
            validate_coordinates(None, 88.61)

    def test_missing_longitude(self):
        with pytest.raises(CoordinateError, match="Missing"):
            validate_coordinates(27.34, None)

    def test_both_missing(self):
        with pytest.raises(CoordinateError, match="Missing"):
            validate_coordinates(None, None)

    def test_latitude_too_high(self):
        with pytest.raises(CoordinateError, match="out of valid range"):
            validate_coordinates(91.0, 88.0)

    def test_latitude_too_low(self):
        with pytest.raises(CoordinateError, match="out of valid range"):
            validate_coordinates(-91.0, 88.0)

    def test_longitude_too_high(self):
        with pytest.raises(CoordinateError, match="out of valid range"):
            validate_coordinates(27.0, 181.0)

    def test_longitude_too_low(self):
        with pytest.raises(CoordinateError, match="out of valid range"):
            validate_coordinates(27.0, -181.0)

    def test_non_numeric_latitude(self):
        with pytest.raises(CoordinateError, match="Non-numeric"):
            validate_coordinates("abc", 88.0)

    def test_null_island_default_allowed(self):
        lat, lon = validate_coordinates(0.0, 0.0)
        assert lat == 0.0 and lon == 0.0

    def test_null_island_rejection(self):
        with pytest.raises(CoordinateError, match="Null Island"):
            validate_coordinates(0.0, 0.0, reject_null_island=True)

    def test_string_numeric_coords(self):
        """String representations of valid numbers should be accepted."""
        lat, lon = validate_coordinates("27.34", "88.61")
        assert lat == pytest.approx(27.34)


class TestIsValidCoordinatePair:
    def test_valid_returns_true(self):
        assert is_valid_coordinate_pair(27.0, 88.0) is True

    def test_invalid_returns_false(self):
        assert is_valid_coordinate_pair(999.0, 88.0) is False

    def test_missing_returns_false(self):
        assert is_valid_coordinate_pair(None, 88.0) is False
