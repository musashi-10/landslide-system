"""
Spatial grid creation and point assignment.

Builds a regular latitude/longitude grid over a bounding box.  Grid
resolution is fully configurable — the same grid can be shared across
the GIS, ML, and data engineering modules via the :class:`GridConfig`
dataclass so all teams use the same spatial identifiers.

Grid cell IDs follow the format::

    GRID_{row:05d}_{col:05d}

where ``row`` and ``col`` are 0-indexed from the south-west corner.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box

logger = logging.getLogger(__name__)


@dataclass
class GridConfig:
    """
    Configuration for a regular spatial grid.

    Attributes
    ----------
    min_lat, max_lat : float
        Latitude extent of the grid in decimal degrees.
    min_lon, max_lon : float
        Longitude extent of the grid in decimal degrees.
    resolution_deg : float
        Grid cell size in decimal degrees (e.g. ``0.01`` ≈ 1 km at equator).
    crs : str
        Coordinate reference system for the grid (default ``"EPSG:4326"``).
    """

    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float
    resolution_deg: float = 0.01
    crs: str = "EPSG:4326"

    def __post_init__(self) -> None:
        if self.min_lat >= self.max_lat:
            raise ValueError(f"min_lat ({self.min_lat}) must be < max_lat ({self.max_lat})")
        if self.min_lon >= self.max_lon:
            raise ValueError(f"min_lon ({self.min_lon}) must be < max_lon ({self.max_lon})")
        if self.resolution_deg <= 0:
            raise ValueError(f"resolution_deg must be positive, got {self.resolution_deg}")


def create_spatial_grid(config: GridConfig) -> gpd.GeoDataFrame:
    """
    Create a regular rectangular grid of polygons over the bounding box
    defined by *config*.

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame with columns:

        * ``grid_id``   – unique cell identifier
        * ``row``       – 0-indexed row from south
        * ``col``       – 0-indexed column from west
        * ``center_lat`` / ``center_lon`` – cell centroid
        * ``geometry``  – Shapely box polygon

    Notes
    -----
    Resolution is configured via :class:`GridConfig`.  DO NOT hard-code a
    single resolution; always pass a ``GridConfig`` instance.
    """
    lats = np.arange(config.min_lat, config.max_lat, config.resolution_deg)
    lons = np.arange(config.min_lon, config.max_lon, config.resolution_deg)

    records = []
    for row_idx, lat in enumerate(lats):
        for col_idx, lon in enumerate(lons):
            lat_max = min(lat + config.resolution_deg, config.max_lat)
            lon_max = min(lon + config.resolution_deg, config.max_lon)
            polygon = box(lon, lat, lon_max, lat_max)
            records.append(
                {
                    "grid_id": f"GRID_{row_idx:05d}_{col_idx:05d}",
                    "row": row_idx,
                    "col": col_idx,
                    "center_lat": (lat + lat_max) / 2,
                    "center_lon": (lon + lon_max) / 2,
                    "geometry": polygon,
                }
            )

    gdf = gpd.GeoDataFrame(records, crs=config.crs)
    logger.info(
        "Created spatial grid: %d cells (res=%.4f°)", len(gdf), config.resolution_deg
    )
    return gdf


def assign_grid_id(
    df: pd.DataFrame,
    config: GridConfig,
    *,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
) -> pd.DataFrame:
    """
    Assign a ``grid_id`` to each row in *df* based on its coordinates.

    Points outside the grid extent receive ``grid_id = None``.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with coordinate columns.
    config : GridConfig
        Grid configuration (must match the grid used by other engineers).
    lat_col, lon_col : str
        Column names for latitude and longitude.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with a ``grid_id`` column added.
    """
    result = df.copy()
    res = config.resolution_deg

    def _compute_grid_id(row):
        lat = row[lat_col]
        lon = row[lon_col]
        if pd.isna(lat) or pd.isna(lon):
            return None
        if not (config.min_lat <= lat < config.max_lat):
            return None
        if not (config.min_lon <= lon < config.max_lon):
            return None
        row_idx = int((lat - config.min_lat) / res)
        col_idx = int((lon - config.min_lon) / res)
        return f"GRID_{row_idx:05d}_{col_idx:05d}"

    result["grid_id"] = result.apply(_compute_grid_id, axis=1)
    assigned = int(result["grid_id"].notna().sum())
    logger.info("Assigned grid_id to %d/%d records.", assigned, len(result))
    return result
