"""
environment/preprocessing/spatial_mapper.py

Spatial mapping: provider grid coordinates -> project location_id.

Weather API providers (Open-Meteo, ERA5, etc.) return data at their internal
grid point nearest to the requested coordinates.  The provider's actual grid
point may differ slightly from the requested location.

This module provides:
1. ``generate_location_id`` — deterministic location_id from lat/lon
   (reuses Engineer 1's format).
2. ``SpatialMapper`` — maps between provider grid coordinates and project
   location_ids, supporting nearest-neighbour lookup when a project location
   list is available.

Spatial mapping method
----------------------
When a list of project locations is provided, the mapper uses a simple
nearest-neighbour search (Haversine distance).  This is appropriate when:
  - The provider grid is finer than the project grid (use nearest project loc)
  - The provider grid is coarser (acceptable error; document grid resolution)

For the prototype, this nearest-neighbour approach is sufficient.
For production, bilinear interpolation or area-weighted averaging could be
applied — document the chosen method.

Coordinate system
-----------------
All coordinates are WGS-84 (EPSG:4326) decimal degrees.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ProjectLocation:
    """A project location entry (from Engineer 1's spatial grid or inventory)."""

    location_id: str
    latitude: float
    longitude: float


class SpatialMapper:
    """
    Map provider coordinates to project location_ids.

    Parameters
    ----------
    project_locations : list[ProjectLocation] | None
        The project's known locations.  If None, the mapper generates a
        location_id directly from the input coordinates (pass-through mode).
    max_distance_km : float
        Maximum allowed distance (km) for a nearest-neighbour match.
        If the nearest project location is farther than this, the mapper
        uses the requested coordinates directly.
    """

    def __init__(
        self,
        project_locations: Optional[List[ProjectLocation]] = None,
        max_distance_km: float = 5.0,
    ) -> None:
        self._locations = project_locations or []
        self._max_distance_km = max_distance_km
        logger.debug(
            "SpatialMapper initialised with %d project locations, max_distance=%.1f km",
            len(self._locations), max_distance_km,
        )

    def map(self, latitude: float, longitude: float) -> Tuple[str, float, float]:
        """
        Return the best matching (location_id, latitude, longitude) for the given coordinates.

        If project locations are provided, finds the nearest match within
        ``max_distance_km``.  Otherwise, generates a location_id directly.

        Parameters
        ----------
        latitude : float
            Provider coordinate latitude.
        longitude : float
            Provider coordinate longitude.

        Returns
        -------
        tuple[str, float, float]
            (location_id, snapped_latitude, snapped_longitude)

        Notes
        -----
        The returned latitude/longitude are the *project* coordinates, not the
        provider grid point.  This ensures the output is consistent with
        Engineer 1's spatial grid.
        """
        if not self._locations:
            loc_id = generate_location_id(latitude, longitude)
            return loc_id, latitude, longitude

        best, dist = self._nearest(latitude, longitude)
        if dist <= self._max_distance_km:
            logger.debug(
                "Nearest project location %s at %.3f km for (%.4f, %.4f)",
                best.location_id, dist, latitude, longitude,
            )
            return best.location_id, best.latitude, best.longitude

        logger.debug(
            "No project location within %.1f km of (%.4f, %.4f) — using direct location_id",
            self._max_distance_km, latitude, longitude,
        )
        loc_id = generate_location_id(latitude, longitude)
        return loc_id, latitude, longitude

    def _nearest(self, lat: float, lon: float) -> Tuple[ProjectLocation, float]:
        """Return the nearest project location and its distance in km."""
        best: Optional[ProjectLocation] = None
        best_dist = float("inf")
        for loc in self._locations:
            d = haversine_km(lat, lon, loc.latitude, loc.longitude)
            if d < best_dist:
                best_dist = d
                best = loc
        return best, best_dist  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Location ID generation (must be compatible with Engineer 1)
# ---------------------------------------------------------------------------


def generate_location_id(latitude: float, longitude: float) -> str:
    """
    Generate a deterministic location_id from coordinates.

    Format matches Engineer 1's ``data_engineering.spatial.location_id``:
        LOC_{lat:+.4f}_{lon:+08.4f}

    Examples
    --------
    >>> generate_location_id(27.123, 88.456)
    'LOC_+27.1230_+088.4560'
    >>> generate_location_id(-34.5678, -58.1234)
    'LOC_-34.5678_-058.1234'
    """
    lat_str = f"{latitude:+.4f}"
    lon_str = f"{longitude:+08.4f}"
    return f"LOC_{lat_str}_{lon_str}"


# ---------------------------------------------------------------------------
# Geometry utilities
# ---------------------------------------------------------------------------


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Compute the great-circle distance between two points (Haversine formula).

    Parameters
    ----------
    lat1, lon1 : float
        First point (WGS-84 decimal degrees).
    lat2, lon2 : float
        Second point (WGS-84 decimal degrees).

    Returns
    -------
    float
        Distance in kilometres.
    """
    R = 6371.0  # Earth radius (km)
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
