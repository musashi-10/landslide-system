"""
Feature extraction pipeline.

Converts raw spatial layers (DEM, soil, geology, land-cover, drainage,
historical landslides) into a single flat feature table aligned to the
spatial grid.

Output columns
--------------
location_id       : str  – primary join key
latitude          : float
longitude         : float
geometry          : str  – WKT of the grid cell polygon
elevation         : float|NaN – metres
slope             : float|NaN – degrees (0–90)
aspect            : float|NaN – degrees clockwise from N (0–360; -1=flat)
soil_type         : str|None
geology           : str|None
land_cover        : str|None
drainage_feature  : int  – 1 if drainage feature present in cell, else 0
historical_landslide : int – count of historical landslide points in cell

Temporal assumption
-------------------
``historical_landslide`` uses ALL records in the provided inventory.
At ML validation time, the ML engineer (Eng-5) is responsible for
filtering this to events that pre-date the prediction horizon.
This avoids leaking future slide events into static features.

Satellite integration
---------------------
Satellite-derived features produced by Engineer 3 can be added later by
joining on ``location_id`` (and optionally ``timestamp_utc``).
This pipeline does NOT fail if satellite data is absent.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd

from gis.config import GISSettings
from gis.dem.processor import open_dem, compute_slope, compute_aspect
from gis.spatial.alignment import (
    reproject_vector,
    spatial_join_to_grid,
    point_to_grid_assignment,
    sample_raster_at_points,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Individual layer loaders
# --------------------------------------------------------------------------- #

def _load_vector_layer(
    path: str,
    target_crs: str,
    label: str,
) -> Optional[gpd.GeoDataFrame]:
    """Open a vector file and reproject to *target_crs*; return None if empty path."""
    if not path:
        logger.info("No path configured for %s layer; skipping.", label)
        return None
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        raise ValueError(f"{label} file {path!r} has no CRS.  Set CRS in the file.")
    return reproject_vector(gdf, target_crs)


# --------------------------------------------------------------------------- #
# Main feature extraction
# --------------------------------------------------------------------------- #

def extract_features(
    grid: gpd.GeoDataFrame,
    settings: GISSettings,
) -> pd.DataFrame:
    """
    Extract all static features onto *grid* using paths in *settings*.

    Parameters
    ----------
    grid     : GeoDataFrame with at least {location_id, latitude, longitude, geometry}.
    settings : GISSettings instance.

    Returns
    -------
    DataFrame with columns listed in the module docstring.
    Missing features are represented as NaN (numeric) or None (categorical).
    """
    crs = settings.target_crs
    df = grid[["location_id", "latitude", "longitude", "geometry"]].copy()
    df["geometry"] = df["geometry"].apply(lambda g: g.wkt)

    lats = df["latitude"].values
    lons = df["longitude"].values

    # ------------------------------------------------------------------ #
    # Terrain: elevation, slope, aspect
    # ------------------------------------------------------------------ #
    if settings.dem_path:
        try:
            elev_data, transform, src_crs = open_dem(settings.dem_path, target_crs=crs)
            slope_data = compute_slope(elev_data, transform, src_crs)
            aspect_data = compute_aspect(elev_data, transform, src_crs)

            from rasterio.transform import Affine
            df["elevation"] = sample_raster_at_points(elev_data, transform, lats, lons)
            df["slope"]     = sample_raster_at_points(slope_data, transform, lats, lons)
            df["aspect"]    = sample_raster_at_points(aspect_data, transform, lats, lons)
        except Exception as exc:
            logger.warning("DEM processing failed (%s); terrain columns set to NaN.", exc)
            df["elevation"] = np.nan
            df["slope"]     = np.nan
            df["aspect"]    = np.nan
    else:
        logger.info("No DEM configured; terrain columns set to NaN.")
        df["elevation"] = np.nan
        df["slope"]     = np.nan
        df["aspect"]    = np.nan

    # ------------------------------------------------------------------ #
    # Soil type
    # ------------------------------------------------------------------ #
    soil_col = "soil_type"
    soil_gdf = _load_vector_layer(settings.soil_path, crs, "soil")
    if soil_gdf is not None and soil_col in soil_gdf.columns:
        grid_with_soil = spatial_join_to_grid(
            grid.to_crs(crs), soil_gdf.to_crs(crs), soil_col
        )
        df = df.merge(
            grid_with_soil[["location_id", soil_col]], on="location_id", how="left"
        )
    else:
        df[soil_col] = None

    # ------------------------------------------------------------------ #
    # Geology
    # ------------------------------------------------------------------ #
    geo_col = "geology"
    geo_gdf = _load_vector_layer(settings.geology_path, crs, "geology")
    if geo_gdf is not None and geo_col in geo_gdf.columns:
        grid_with_geo = spatial_join_to_grid(
            grid.to_crs(crs), geo_gdf.to_crs(crs), geo_col
        )
        df = df.merge(
            grid_with_geo[["location_id", geo_col]], on="location_id", how="left"
        )
    else:
        df[geo_col] = None

    # ------------------------------------------------------------------ #
    # Land cover
    # ------------------------------------------------------------------ #
    lc_col = "land_cover"
    lc_gdf = _load_vector_layer(settings.landcover_path, crs, "land_cover")
    if lc_gdf is not None and lc_col in lc_gdf.columns:
        grid_with_lc = spatial_join_to_grid(
            grid.to_crs(crs), lc_gdf.to_crs(crs), lc_col
        )
        df = df.merge(
            grid_with_lc[["location_id", lc_col]], on="location_id", how="left"
        )
    else:
        df[lc_col] = None

    # ------------------------------------------------------------------ #
    # Drainage
    # ------------------------------------------------------------------ #
    dr_gdf = _load_vector_layer(settings.drainage_path, crs, "drainage")
    if dr_gdf is not None:
        drain_count_col = "drainage_feature"
        # Treat any intersecting drainage geometry as presence=1
        grid_tmp = grid.to_crs(crs)[["location_id", "geometry"]]
        joined = gpd.sjoin(
            grid_tmp,
            dr_gdf[["geometry"]].to_crs(crs).assign(_d=1),
            how="left",
            predicate="intersects",
        )
        counts = joined.groupby("location_id")["_d"].sum().reset_index()
        counts.columns = ["location_id", drain_count_col]
        counts[drain_count_col] = (counts[drain_count_col] > 0).astype(int)
        df = df.merge(counts, on="location_id", how="left")
        df[drain_count_col] = df[drain_count_col].fillna(0).astype(int)
    else:
        df["drainage_feature"] = 0

    # ------------------------------------------------------------------ #
    # Historical landslides
    # Temporal note: the caller is responsible for pre-filtering the
    # inventory to events that pre-date the prediction horizon.
    # ------------------------------------------------------------------ #
    hist_gdf = _load_vector_layer(settings.historical_path, crs, "historical_landslide")
    if hist_gdf is not None:
        grid_tmp = grid.to_crs(crs)[["location_id", "geometry"]].copy()
        hist_pts = hist_gdf.to_crs(crs).copy()
        # If polygon inventory, convert to centroid points
        if not hist_pts.geometry.geom_type.isin(["Point"]).all():
            hist_pts["geometry"] = hist_pts.geometry.centroid
        result_with_hist = point_to_grid_assignment(
            grid_tmp, hist_pts, count_col="historical_landslide"
        )
        df = df.merge(
            result_with_hist[["location_id", "historical_landslide"]],
            on="location_id",
            how="left",
        )
        df["historical_landslide"] = df["historical_landslide"].fillna(0).astype(int)
    else:
        df["historical_landslide"] = 0

    # ------------------------------------------------------------------ #
    # Enforce column order per data-contract
    # ------------------------------------------------------------------ #
    ordered_cols = [
        "location_id", "latitude", "longitude", "geometry",
        "elevation", "slope", "aspect",
        "soil_type", "geology", "land_cover",
        "drainage_feature", "historical_landslide",
    ]
    df = df[ordered_cols]
    logger.info(
        "Feature extraction complete: %d locations, columns=%s",
        len(df), list(df.columns),
    )
    return df
