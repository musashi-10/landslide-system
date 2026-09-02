"""
General spatial operation utilities.

Provides reusable wrappers around geopandas operations:

* :func:`points_to_geodataframe`  — build a GeoDataFrame from lat/lon columns
* :func:`spatial_join`            — join attributes from a polygon layer to points
* :func:`clip_to_region`          — clip a GeoDataFrame to a study-area polygon
* :func:`transform_crs`           — reproject a GeoDataFrame
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

logger = logging.getLogger(__name__)

DEFAULT_CRS = "EPSG:4326"


def points_to_geodataframe(
    df: pd.DataFrame,
    *,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    crs: str = DEFAULT_CRS,
) -> gpd.GeoDataFrame:
    """
    Convert a DataFrame with lat/lon columns into a point GeoDataFrame.

    Rows with missing or non-numeric coordinates are dropped with a warning.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    lat_col, lon_col : str
        Coordinate column names.
    crs : str
        CRS for the output GeoDataFrame.

    Returns
    -------
    geopandas.GeoDataFrame
        Point GeoDataFrame.
    """
    valid_mask = df[lat_col].notna() & df[lon_col].notna()
    n_dropped = int((~valid_mask).sum())
    if n_dropped:
        logger.warning(
            "Dropped %d rows with missing coordinates when building GeoDataFrame.",
            n_dropped,
        )

    valid_df = df.loc[valid_mask].copy()
    geometry = [
        Point(row[lon_col], row[lat_col]) for _, row in valid_df.iterrows()
    ]
    gdf = gpd.GeoDataFrame(valid_df, geometry=geometry, crs=crs)
    logger.info("Created GeoDataFrame with %d point features.", len(gdf))
    return gdf


def spatial_join(
    points_gdf: gpd.GeoDataFrame,
    polygons_gdf: gpd.GeoDataFrame,
    *,
    how: str = "left",
    predicate: str = "within",
    rsuffix: str = "_poly",
) -> gpd.GeoDataFrame:
    """
    Perform a spatial join from *points_gdf* to *polygons_gdf*.

    Parameters
    ----------
    points_gdf : gpd.GeoDataFrame
        Point layer (left side of join).
    polygons_gdf : gpd.GeoDataFrame
        Polygon layer (right side).
    how : str
        Join type: ``"left"``, ``"right"``, or ``"inner"``.
    predicate : str
        Spatial predicate (default ``"within"``).
    rsuffix : str
        Suffix for conflicting column names from the right GeoDataFrame.

    Returns
    -------
    geopandas.GeoDataFrame
        Joined GeoDataFrame.
    """
    # Ensure matching CRS
    if points_gdf.crs != polygons_gdf.crs:
        logger.info(
            "CRS mismatch in spatial_join; reprojecting polygons to %s.",
            points_gdf.crs,
        )
        polygons_gdf = polygons_gdf.to_crs(points_gdf.crs)

    result = gpd.sjoin(
        points_gdf, polygons_gdf, how=how, predicate=predicate, rsuffix=rsuffix
    )
    logger.info("Spatial join produced %d records.", len(result))
    return result


def clip_to_region(
    gdf: gpd.GeoDataFrame,
    region: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Clip *gdf* to the bounding geometry of *region*.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Data to clip.
    region : gpd.GeoDataFrame
        Study-region polygon(s).

    Returns
    -------
    geopandas.GeoDataFrame
        Clipped GeoDataFrame.
    """
    if gdf.crs != region.crs:
        logger.info("Reprojecting region to match data CRS (%s).", gdf.crs)
        region = region.to_crs(gdf.crs)

    clipped = gpd.clip(gdf, region)
    logger.info(
        "Clipped from %d to %d features.", len(gdf), len(clipped)
    )
    return clipped


def transform_crs(
    gdf: gpd.GeoDataFrame,
    target_crs: str,
) -> gpd.GeoDataFrame:
    """
    Reproject *gdf* to *target_crs*.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Input GeoDataFrame.
    target_crs : str
        Target CRS string (e.g. ``"EPSG:32644"``).

    Returns
    -------
    geopandas.GeoDataFrame
        Reprojected GeoDataFrame.
    """
    if gdf.crs is None:
        raise ValueError("Input GeoDataFrame has no CRS set.")
    result = gdf.to_crs(target_crs)
    logger.info("Transformed CRS: %s → %s", gdf.crs, target_crs)
    return result
