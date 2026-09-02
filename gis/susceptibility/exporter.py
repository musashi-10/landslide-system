"""
Raster export for the susceptibility layer.

Writes a GeoTIFF from a 1-D susceptibility Series aligned to a grid.
Also produces a GeoPackage for vector visualization.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds

logger = logging.getLogger(__name__)


def export_susceptibility_raster(
    features: pd.DataFrame,
    susceptibility: pd.Series,
    output_path: str | Path,
    resolution: float = 0.01,
    crs_str: str = "EPSG:4326",
    nodata: float = -9999.0,
) -> None:
    """
    Write a GeoTIFF raster of the susceptibility scores.

    Parameters
    ----------
    features      : Feature DataFrame with latitude, longitude columns.
    susceptibility: pd.Series aligned to features.index.
    output_path   : Output .tif path.
    resolution    : Cell size (degrees for EPSG:4326).
    crs_str       : CRS EPSG string.
    nodata        : Fill value for NaN cells.
    """
    merged = features[["latitude", "longitude"]].copy()
    merged["susceptibility"] = susceptibility.values

    lats = np.sort(merged["latitude"].unique())[::-1]   # north → south
    lons = np.sort(merged["longitude"].unique())         # west  → east

    n_rows = len(lats)
    n_cols = len(lons)

    if n_rows == 0 or n_cols == 0:
        logger.warning("No data to export to raster.")
        return

    # Build lookup
    lat_idx = {v: i for i, v in enumerate(lats)}
    lon_idx = {v: i for i, v in enumerate(lons)}

    raster = np.full((n_rows, n_cols), nodata, dtype=np.float32)
    for _, row in merged.iterrows():
        r = lat_idx.get(row["latitude"])
        c = lon_idx.get(row["longitude"])
        if r is not None and c is not None:
            val = row["susceptibility"]
            raster[r, c] = nodata if pd.isna(val) else float(val)

    west  = lons.min()
    east  = lons.max() + resolution
    south = lats.min()
    north = lats.max() + resolution
    transform = from_bounds(west, south, east, north, n_cols, n_rows)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=n_rows,
        width=n_cols,
        count=1,
        dtype="float32",
        crs=CRS.from_string(crs_str),
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(raster, 1)
    logger.info("Susceptibility raster saved: %s (%d×%d)", output_path, n_rows, n_cols)


def export_susceptibility_vector(
    features: pd.DataFrame,
    susceptibility: pd.Series,
    output_path: str | Path,
    crs_str: str = "EPSG:4326",
) -> None:
    """
    Export susceptibility as a GeoPackage (polygon layer) for visualization.

    Parameters
    ----------
    features      : Feature DataFrame with geometry (WKT) and location_id.
    susceptibility: pd.Series aligned to features.index.
    output_path   : Output .gpkg path.
    crs_str       : CRS EPSG string.
    """
    from shapely import wkt as shapely_wkt

    df = features[["location_id", "latitude", "longitude", "geometry"]].copy()
    df["susceptibility"] = susceptibility.values
    df["geometry"] = df["geometry"].apply(
        lambda g: shapely_wkt.loads(g) if isinstance(g, str) else g
    )
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs=crs_str)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(output_path, driver="GPKG")
    logger.info("Susceptibility vector saved: %s (%d features)", output_path, len(gdf))
