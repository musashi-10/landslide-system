"""
Spatial alignment utilities.

These functions bring rasters and vectors into a common coordinate frame
so they can be safely combined without silently mixing CRS.

Strategy
--------
1. All datasets are reprojected to the shared ``target_crs`` (EPSG:4326 by
   default).
2. Rasters are aligned to the DEM grid via ``align_rasters`` which matches
   resolution and extent.
3. Vector layers are spatially joined to the grid centroids.
4. Resampling is performed at most once per input layer.

IMPORTANT: Do not call align_rasters repeatedly on the same source – pass
the already-aligned result forward.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import reproject as rasterio_reproject
from shapely.geometry import Point

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Vector helpers
# --------------------------------------------------------------------------- #

def reproject_vector(
    gdf: gpd.GeoDataFrame,
    target_crs: str = "EPSG:4326",
) -> gpd.GeoDataFrame:
    """
    Reproject a GeoDataFrame to *target_crs*.

    If the GDF has no CRS set, a ``ValueError`` is raised (never assume CRS).

    Parameters
    ----------
    gdf        : Input GeoDataFrame.
    target_crs : EPSG string.

    Returns
    -------
    Reprojected GeoDataFrame (copy if CRS changed, same object if already correct).
    """
    if gdf.crs is None:
        raise ValueError(
            "GeoDataFrame has no CRS.  Set CRS explicitly before passing to "
            "reproject_vector()."
        )
    target = CRS.from_string(target_crs)
    if gdf.crs == target:
        return gdf
    logger.debug("Reprojecting vector %s → %s", gdf.crs, target_crs)
    return gdf.to_crs(target_crs)


def spatial_join_to_grid(
    grid: gpd.GeoDataFrame,
    layer: gpd.GeoDataFrame,
    attribute_col: str,
    how: str = "left",
    predicate: str = "intersects",
) -> gpd.GeoDataFrame:
    """
    Join a single attribute from *layer* onto *grid* cells.

    For grid cells that match multiple features the *first* match is kept.

    Parameters
    ----------
    grid          : Grid GeoDataFrame (must have ``location_id`` column).
    layer         : Vector layer to join (must share CRS with grid).
    attribute_col : Column name in *layer* to copy onto the grid.
    how           : sjoin how argument ('left', 'inner').
    predicate     : sjoin predicate ('intersects', 'contains', 'within').

    Returns
    -------
    Grid GeoDataFrame with *attribute_col* added.
    """
    if grid.crs != layer.crs:
        raise ValueError(
            f"CRS mismatch: grid={grid.crs}, layer={layer.crs}.  "
            "Reproject with reproject_vector() first."
        )
    if attribute_col not in layer.columns:
        raise ValueError(f"Column '{attribute_col}' not found in layer.")

    joined = gpd.sjoin(
        grid[["location_id", "geometry"]],
        layer[["geometry", attribute_col]],
        how=how,
        predicate=predicate,
    )
    # Keep first match per cell
    joined = joined.drop_duplicates(subset="location_id")
    result = grid.merge(
        joined[["location_id", attribute_col]],
        on="location_id",
        how="left",
    )
    logger.debug(
        "Spatial join '%s': %d/%d grid cells matched",
        attribute_col,
        result[attribute_col].notna().sum(),
        len(result),
    )
    return result


def point_to_grid_assignment(
    grid: gpd.GeoDataFrame,
    points: gpd.GeoDataFrame,
    count_col: str = "point_count",
) -> gpd.GeoDataFrame:
    """
    Count points falling within each grid cell.

    Useful for historical landslide counts.

    Parameters
    ----------
    grid      : Grid GeoDataFrame.
    points    : Point GeoDataFrame (same CRS as grid).
    count_col : Output column name for the count.

    Returns
    -------
    Grid GeoDataFrame with *count_col* added (0 where no points).
    """
    if grid.crs != points.crs:
        raise ValueError("CRS mismatch between grid and points.")

    joined = gpd.sjoin(
        points[["geometry"]].assign(_pt=1),
        grid[["location_id", "geometry"]],
        how="left",
        predicate="within",
    )
    counts = joined.groupby("location_id")["_pt"].sum().reset_index()
    counts.columns = ["location_id", count_col]
    result = grid.merge(counts, on="location_id", how="left")
    result[count_col] = result[count_col].fillna(0).astype(int)
    return result


# --------------------------------------------------------------------------- #
# Raster helpers
# --------------------------------------------------------------------------- #

def align_rasters(
    source_data: np.ndarray,
    source_transform: Affine,
    source_crs: CRS,
    target_transform: Affine,
    target_crs: CRS,
    target_shape: Tuple[int, int],
    resampling: Resampling = Resampling.bilinear,
) -> np.ndarray:
    """
    Align *source_data* to the *target* raster grid (extent + resolution).

    This is the canonical function to bring any raster into the DEM grid
    before combining.

    Parameters
    ----------
    source_data      : float32 ndarray, nodata as NaN.
    source_transform : Affine of source.
    source_crs       : CRS of source.
    target_transform : Affine of target (reference) grid.
    target_crs       : CRS of target grid.
    target_shape     : (rows, cols) of the target grid.
    resampling       : Resampling algorithm.

    Returns
    -------
    Aligned float32 ndarray with the same shape as *target_shape*.
    """
    out = np.full(target_shape, np.nan, dtype=np.float32)
    rasterio_reproject(
        source=source_data,
        destination=out,
        src_transform=source_transform,
        src_crs=source_crs,
        dst_transform=target_transform,
        dst_crs=target_crs,
        resampling=resampling,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )
    return out


def sample_raster_at_points(
    data: np.ndarray,
    transform: Affine,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
) -> np.ndarray:
    """
    Sample raster values at (lat, lon) point arrays.

    Parameters
    ----------
    data        : float32 ndarray (rows × cols).
    transform   : Affine of the raster (EPSG:4326 assumed).
    latitudes   : 1-D float array of latitudes.
    longitudes  : 1-D float array of longitudes.

    Returns
    -------
    values : float array, same length as latitudes; NaN for out-of-bounds.
    """
    rows, cols = data.shape
    row_idx = ((latitudes  - transform.f) / transform.e).astype(int)
    col_idx = ((longitudes - transform.c) / transform.a).astype(int)

    valid = (
        (row_idx >= 0) & (row_idx < rows) &
        (col_idx >= 0) & (col_idx < cols)
    )
    values = np.full(len(latitudes), np.nan, dtype=np.float32)
    values[valid] = data[row_idx[valid], col_idx[valid]]
    return values
