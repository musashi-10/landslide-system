"""
Spatial grid generation.

Creates a regular rectangular grid of cells over an area of interest and
assigns each cell a unique ``grid_id`` / ``location_id``.

All grid cells are output in the configured ``target_crs`` (default EPSG:4326).

Grid IDs
--------
GRID_00001, GRID_00002, …  (zero-padded to 5 digits; extend as needed)

Every cell record contains:
  grid_id    : str   – e.g. "GRID_00001"
  location_id: str   – same as grid_id (primary join key for other modules)
  geometry   : str   – WKT polygon of the cell boundary
  latitude   : float – centroid latitude  (degrees, WGS-84 if EPSG:4326)
  longitude  : float – centroid longitude (degrees)
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import geopandas as gpd
import numpy as np
from shapely.geometry import box

logger = logging.getLogger(__name__)


def generate_grid(
    bbox: Tuple[float, float, float, float],  # (west, south, east, north)
    resolution: float,
    crs: str = "EPSG:4326",
    id_prefix: str = "GRID",
) -> gpd.GeoDataFrame:
    """
    Generate a regular grid of rectangular cells over *bbox*.

    Parameters
    ----------
    bbox       : (west, south, east, north) in *crs* units.
    resolution : Cell side length in *crs* units (degrees for EPSG:4326).
    crs        : EPSG string for the output grid.
    id_prefix  : Prefix for grid / location IDs (default "GRID").

    Returns
    -------
    GeoDataFrame with columns:
        grid_id, location_id, geometry, latitude, longitude

    Raises
    ------
    ValueError : If bbox is degenerate or resolution is non-positive.
    """
    west, south, east, north = bbox
    if west >= east or south >= north:
        raise ValueError(
            f"Degenerate bbox: west={west}, south={south}, east={east}, north={north}. "
            "west must be < east and south must be < north."
        )
    if resolution <= 0:
        raise ValueError(f"resolution must be > 0, got {resolution}.")

    num_cols = int(np.ceil(round((east - west) / resolution, 7)))
    num_rows = int(np.ceil(round((north - south) / resolution, 7)))
    cols = west + np.arange(num_cols) * resolution
    rows = south + np.arange(num_rows) * resolution

    records: List[dict] = []
    idx = 1
    for row_south in rows:
        for col_west in cols:
            cell = box(
                col_west,
                row_south,
                min(col_west + resolution, east),
                min(row_south + resolution, north),
            )
            centroid = cell.centroid
            cell_id = f"{id_prefix}_{idx:05d}"
            records.append(
                {
                    "grid_id":     cell_id,
                    "location_id": cell_id,
                    "geometry":    cell,
                    "latitude":    centroid.y,
                    "longitude":   centroid.x,
                }
            )
            idx += 1

    if not records:
        raise ValueError(
            f"No grid cells generated for bbox={bbox} and resolution={resolution}."
        )

    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=crs)
    logger.info(
        "Generated grid: %d cells, resolution=%s, CRS=%s, bbox=%s",
        len(gdf), resolution, crs, bbox,
    )
    return gdf
