"""
Coordinate validation utilities.

WGS-84 bounds:
  latitude  : [-90, 90]
  longitude : [-180, 180]

Coordinates of exactly (0.0, 0.0) are considered suspicious (Null Island)
but are *not* rejected automatically — they are flagged in the report.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class CoordinateError(ValueError):
    """Raised when a coordinate value is invalid."""


def validate_coordinates(
    latitude: Optional[float],
    longitude: Optional[float],
    *,
    reject_null_island: bool = False,
) -> tuple[float, float]:
    """
    Validate and return a ``(latitude, longitude)`` pair.

    Parameters
    ----------
    latitude : float | None
        Latitude value to validate.
    longitude : float | None
        Longitude value to validate.
    reject_null_island : bool
        If ``True``, treat exact ``(0.0, 0.0)`` as invalid.

    Returns
    -------
    tuple[float, float]
        The validated ``(latitude, longitude)`` pair.

    Raises
    ------
    CoordinateError
        When either coordinate is None, non-numeric, or out of range.
    """
    if latitude is None or longitude is None:
        raise CoordinateError(
            f"Missing coordinate: latitude={latitude!r}, longitude={longitude!r}"
        )

    try:
        lat = float(latitude)
        lon = float(longitude)
    except (TypeError, ValueError) as exc:
        raise CoordinateError(
            f"Non-numeric coordinate: latitude={latitude!r}, longitude={longitude!r}"
        ) from exc

    if not (-90.0 <= lat <= 90.0):
        raise CoordinateError(
            f"Latitude {lat} out of valid range [-90, 90]"
        )

    if not (-180.0 <= lon <= 180.0):
        raise CoordinateError(
            f"Longitude {lon} out of valid range [-180, 180]"
        )

    if reject_null_island and lat == 0.0 and lon == 0.0:
        raise CoordinateError(
            "Coordinate (0.0, 0.0) rejected: likely erroneous Null Island value"
        )

    return lat, lon


def is_valid_coordinate_pair(
    latitude: Optional[float],
    longitude: Optional[float],
    *,
    reject_null_island: bool = False,
) -> bool:
    """
    Return ``True`` if the coordinate pair is valid, ``False`` otherwise.

    This is a non-raising convenience wrapper around :func:`validate_coordinates`.
    """
    try:
        validate_coordinates(latitude, longitude, reject_null_island=reject_null_island)
        return True
    except CoordinateError:
        return False
