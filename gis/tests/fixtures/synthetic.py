"""
Synthetic GIS test fixtures.

Creates tiny in-memory or on-disk raster/vector datasets that allow the
entire GIS pipeline to run without downloading any external data.

Usage (in tests):
    from gis.tests.fixtures.synthetic import (
        make_dem_raster,
        make_soil_vector,
        make_historical_landslides,
    )
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Tuple

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from shapely.geometry import Point, Polygon, box


# --------------------------------------------------------------------------- #
# Raster fixtures
# --------------------------------------------------------------------------- #

def make_dem_array(
    rows: int = 20,
    cols: int = 20,
    elevation_base: float = 1000.0,
    elevation_range: float = 800.0,
    seed: int = 42,
) -> Tuple[np.ndarray, rasterio.transform.Affine, CRS]:
    """
    Create a small synthetic DEM array.

    Returns a (rows × cols) float32 array with gradients that will produce
    non-trivial slope and aspect values.  A few pixels are set to NaN to
    exercise nodata propagation.

    CRS: EPSG:4326.
    Extent: 88.00–88.20°E, 27.00–27.20°N (Sikkim, India — known landslide area).
    """
    rng = np.random.default_rng(seed)
    # Create a sloped surface with noise
    row_grad = np.linspace(0, 1, rows)[:, None] * elevation_range
    col_grad = np.linspace(0, 0.5, cols)[None, :] * elevation_range
    noise = rng.normal(0, 20, (rows, cols)).astype(np.float32)
    dem = (elevation_base + row_grad + col_grad + noise).astype(np.float32)

    # Introduce a few NaN cells
    dem[0, 0] = np.nan
    dem[rows - 1, cols - 1] = np.nan

    west, south, east, north = 88.00, 27.00, 88.20, 27.20
    transform = from_bounds(west, south, east, north, cols, rows)
    crs = CRS.from_epsg(4326)
    return dem, transform, crs


def make_dem_file(path: str | Path) -> Path:
    """Write a synthetic DEM to *path* (GeoTIFF) and return the path."""
    dem, transform, crs = make_dem_array()
    data = dem.copy()
    nodata_val = -9999.0
    data[np.isnan(data)] = nodata_val
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w",
        driver="GTiff",
        height=dem.shape[0],
        width=dem.shape[1],
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=nodata_val,
    ) as dst:
        dst.write(data, 1)
    return path


# --------------------------------------------------------------------------- #
# Vector fixtures
# --------------------------------------------------------------------------- #

def make_soil_vector() -> gpd.GeoDataFrame:
    """
    Return a tiny soil polygon GeoDataFrame covering the synthetic DEM extent.

    Soil types: 'clay' (west half) and 'loam' (east half).
    CRS: EPSG:4326.
    """
    west, south, east, north = 88.00, 27.00, 88.20, 27.20
    mid = (west + east) / 2.0
    polygons = [
        box(west, south, mid, north),
        box(mid,  south, east, north),
    ]
    return gpd.GeoDataFrame(
        {"soil_type": ["clay", "loam"], "geometry": polygons},
        crs="EPSG:4326",
    )


def make_geology_vector() -> gpd.GeoDataFrame:
    """
    Return a small geology polygon GeoDataFrame.

    Geology types: 'schist' (north half) and 'granite' (south half).
    """
    west, south, east, north = 88.00, 27.00, 88.20, 27.20
    mid = (south + north) / 2.0
    polygons = [
        box(west, mid,   east, north),
        box(west, south, east, mid),
    ]
    return gpd.GeoDataFrame(
        {"geology": ["schist", "granite"], "geometry": polygons},
        crs="EPSG:4326",
    )


def make_landcover_vector() -> gpd.GeoDataFrame:
    """Return a land-cover polygon GeoDataFrame with 'forest' and 'bare'."""
    west, south, east, north = 88.00, 27.00, 88.20, 27.20
    mid_lon = (west + east)   / 2.0
    mid_lat = (south + north) / 2.0
    polygons = [
        box(west,    south, mid_lon, mid_lat),
        box(mid_lon, south, east,    mid_lat),
        box(west,    mid_lat, mid_lon, north),
        box(mid_lon, mid_lat, east,    north),
    ]
    return gpd.GeoDataFrame(
        {"land_cover": ["forest", "bare", "grassland", "shrub"], "geometry": polygons},
        crs="EPSG:4326",
    )


def make_drainage_vector() -> gpd.GeoDataFrame:
    """Return a simple line GeoDataFrame representing drainage features."""
    from shapely.geometry import LineString
    line = LineString([(88.05, 27.05), (88.10, 27.10), (88.15, 27.05)])
    return gpd.GeoDataFrame(
        {"feature_type": ["stream"], "geometry": [line]},
        crs="EPSG:4326",
    )


def make_historical_landslides(n: int = 8) -> gpd.GeoDataFrame:
    """
    Return a point GeoDataFrame of synthetic historical landslide locations.

    Points are scattered within the synthetic DEM extent.
    """
    rng = np.random.default_rng(99)
    lons = rng.uniform(88.02, 88.18, n)
    lats = rng.uniform(27.02, 27.18, n)
    points = [Point(lo, la) for lo, la in zip(lons, lats)]
    return gpd.GeoDataFrame(
        {"event_id": [f"LS_{i:03d}" for i in range(n)], "geometry": points},
        crs="EPSG:4326",
    )
