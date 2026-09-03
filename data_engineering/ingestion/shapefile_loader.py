"""
Shapefile / GeoPackage loader.

Uses geopandas (backed by Fiona) to read Shapefiles (``.shp``) and
GeoPackages (``.gpkg``).  Reprojects to WGS-84 when necessary.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import geopandas as gpd

logger = logging.getLogger(__name__)

TARGET_CRS = "EPSG:4326"

_SUPPORTED_SUFFIXES = {".shp", ".gpkg", ".geojson", ".json"}


class ShapefileIngestionError(ValueError):
    """Raised when a Shapefile or GeoPackage cannot be ingested."""


def load_shapefile(
    path: str | Path,
    *,
    layer: Optional[str] = None,
    required_columns: Optional[list[str]] = None,
    source_name: Optional[str] = None,
    target_crs: str = TARGET_CRS,
) -> gpd.GeoDataFrame:
    """
    Load a Shapefile or GeoPackage into a GeoDataFrame.

    Parameters
    ----------
    path : str | Path
        Path to ``.shp`` or ``.gpkg`` file.
    layer : str | None
        Layer name for GeoPackages with multiple layers.  Ignored for
        Shapefiles.
    required_columns : list[str] | None
        Attribute column names that must be present.
    source_name : str | None
        Source identifier stored in a ``source`` column.
    target_crs : str
        Output CRS (default ``"EPSG:4326"``).

    Returns
    -------
    geopandas.GeoDataFrame

    Raises
    ------
    ShapefileIngestionError
        When the file is missing, unsupported, or required columns are absent.
    """
    path = Path(path)
    if not path.exists():
        raise ShapefileIngestionError(f"File not found: {path}")

    if path.suffix.lower() not in _SUPPORTED_SUFFIXES:
        raise ShapefileIngestionError(
            f"Unsupported file type {path.suffix!r}. Supported: {_SUPPORTED_SUFFIXES}"
        )

    logger.info("Loading vector file: %s (layer=%s)", path, layer)

    read_kwargs = {}
    if layer:
        read_kwargs["layer"] = layer

    try:
        gdf = gpd.read_file(str(path), **read_kwargs)
    except Exception as exc:
        raise ShapefileIngestionError(f"Failed to read {path}: {exc}") from exc

    if gdf.empty:
        raise ShapefileIngestionError(f"Vector file is empty: {path}")

    # CRS handling
    if gdf.crs is None:
        logger.warning("File %s has no CRS. Assuming %s.", path.name, target_crs)
        gdf = gdf.set_crs(target_crs)
    elif gdf.crs.to_epsg() != 4326:
        logger.info("Reprojecting from %s to %s.", gdf.crs.to_string(), target_crs)
        gdf = gdf.to_crs(target_crs)

    if required_columns:
        missing = [c for c in required_columns if c not in gdf.columns]
        if missing:
            raise ShapefileIngestionError(
                f"{path.name} is missing required columns: {missing}"
            )

    gdf["source"] = source_name or path.name

    logger.info("Loaded %d features from %s", len(gdf), path.name)
    return gdf
