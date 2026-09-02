"""
GeoJSON loader for point/line/polygon landslide datasets.

Reads a GeoJSON file into a GeoDataFrame via geopandas, reprojects to
WGS-84 (EPSG:4326) when necessary, and adds source metadata.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import geopandas as gpd

logger = logging.getLogger(__name__)

TARGET_CRS = "EPSG:4326"


class GeoJSONIngestionError(ValueError):
    """Raised when a GeoJSON file cannot be ingested."""


def load_geojson(
    path: str | Path,
    *,
    required_columns: Optional[list[str]] = None,
    source_name: Optional[str] = None,
    target_crs: str = TARGET_CRS,
) -> gpd.GeoDataFrame:
    """
    Load a GeoJSON file into a GeoDataFrame reprojected to *target_crs*.

    Parameters
    ----------
    path : str | Path
        Path to the ``.geojson`` file.
    required_columns : list[str] | None
        Attribute column names that must be present after loading.
    source_name : str | None
        Source identifier stored in a ``source`` column.
    target_crs : str
        EPSG/proj string for the output CRS (default ``"EPSG:4326"``).

    Returns
    -------
    geopandas.GeoDataFrame
        Loaded and reprojected GeoDataFrame with a ``source`` column.

    Raises
    ------
    GeoJSONIngestionError
        When the file is missing, empty, or required columns are absent.
    """
    path = Path(path)
    if not path.exists():
        raise GeoJSONIngestionError(f"File not found: {path}")

    logger.info("Loading GeoJSON: %s", path)

    try:
        gdf = gpd.read_file(str(path))
    except Exception as exc:
        raise GeoJSONIngestionError(f"Failed to read GeoJSON {path}: {exc}") from exc

    if gdf.empty:
        raise GeoJSONIngestionError(f"GeoJSON file is empty: {path}")

    # CRS detection and reprojection
    if gdf.crs is None:
        logger.warning(
            "GeoJSON %s has no CRS defined. Assuming %s.", path.name, target_crs
        )
        gdf = gdf.set_crs(target_crs)
    elif gdf.crs.to_epsg() != 4326:
        logger.info(
            "Reprojecting %s from %s to %s.", path.name, gdf.crs.to_string(), target_crs
        )
        gdf = gdf.to_crs(target_crs)

    # Column validation
    if required_columns:
        missing = [c for c in required_columns if c not in gdf.columns]
        if missing:
            raise GeoJSONIngestionError(
                f"GeoJSON {path.name} is missing required columns: {missing}"
            )

    gdf["source"] = source_name or path.name

    logger.info("Loaded %d features from %s (CRS: %s)", len(gdf), path.name, gdf.crs)
    return gdf
