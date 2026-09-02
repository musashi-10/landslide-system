"""
Main GIS pipeline — public integration interface for Engineer 5 (ML).

Entry point
-----------
    from gis import build_static_feature_dataset

    feature_parquet, susceptibility_tif = build_static_feature_dataset(
        bbox=(88.0, 27.0, 89.0, 28.0),
        output_dir="gis_outputs",
    )

The function returns paths to:
  1. ``static_features.parquet`` – flat feature table per grid cell
  2. ``susceptibility.tif``       – GeoTIFF baseline susceptibility layer
  3. ``susceptibility.gpkg``      – GeoPackage for vector visualization
  4. ``metadata.json``            – provenance sidecar

Engineer 5 needs only the Parquet file.  The raster/vector outputs are for
visualization and sanity-checking.

Satellite integration
---------------------
Engineer 3 produces satellite-derived features keyed by ``location_id``.
To join them:

    import pandas as pd
    static = pd.read_parquet("gis_outputs/static_features.parquet")
    satellite = pd.read_parquet("satellite_outputs/features.parquet")
    combined = static.merge(satellite, on="location_id", how="left")

The pipeline NEVER fails if satellite data is absent.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from gis.config import GISSettings, get_gis_settings
from gis.spatial.grid import generate_grid
from gis.preprocessing.extractor import extract_features
from gis.susceptibility.baseline import compute_baseline_susceptibility
from gis.susceptibility.exporter import (
    export_susceptibility_raster,
    export_susceptibility_vector,
)
from gis.metadata import build_metadata, save_metadata

logger = logging.getLogger(__name__)


def build_static_feature_dataset(
    bbox: Tuple[float, float, float, float],
    output_dir: Optional[str] = None,
    settings: Optional[GISSettings] = None,
) -> Tuple[Path, Path, Path]:
    """
    Build the complete static GIS feature dataset for an area of interest.

    This is the primary integration API for Engineer 5 (ML Engineer).

    Parameters
    ----------
    bbox       : (west, south, east, north) in EPSG:4326 degrees.
                 Example: (88.0, 27.0, 89.0, 28.0) for eastern Sikkim.
    output_dir : Directory where outputs are written.
                 Defaults to ``settings.output_dir`` (or "gis_outputs").
    settings   : Optional GISSettings override (useful in tests).
                 If None, the cached global settings are used.

    Returns
    -------
    feature_parquet  : Path to static_features.parquet
    susceptibility_tif : Path to susceptibility.tif
    susceptibility_gpkg : Path to susceptibility.gpkg

    Output files
    ------------
    static_features.parquet  – main static feature table (Engineer 5 input)
    susceptibility.tif       – GeoTIFF susceptibility layer
    susceptibility.gpkg      – GeoPackage susceptibility layer
    metadata.json            – provenance metadata

    Columns in static_features.parquet
    -----------------------------------
    location_id, latitude, longitude, geometry,
    elevation, slope, aspect,
    soil_type, geology, land_cover,
    drainage_feature, historical_landslide,
    susceptibility
    """
    if settings is None:
        settings = get_gis_settings()

    out_dir = Path(output_dir or settings.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Step 1 — Generate spatial grid
    # ------------------------------------------------------------------ #
    logger.info("Step 1/5: Generating grid for bbox=%s res=%s", bbox, settings.grid_resolution_deg)
    grid = generate_grid(
        bbox=bbox,
        resolution=settings.grid_resolution_deg,
        crs=settings.target_crs,
    )

    # ------------------------------------------------------------------ #
    # Step 2 — Extract static features
    # ------------------------------------------------------------------ #
    logger.info("Step 2/5: Extracting static features.")
    features = extract_features(grid, settings)

    # ------------------------------------------------------------------ #
    # Step 3 — Compute baseline susceptibility
    # WARNING: weights are not scientifically validated.
    # ------------------------------------------------------------------ #
    logger.info("Step 3/5: Computing baseline susceptibility.")
    susc = compute_baseline_susceptibility(features, settings.susceptibility_weights)
    features["susceptibility"] = susc

    # ------------------------------------------------------------------ #
    # Step 4 — Save outputs
    # ------------------------------------------------------------------ #
    logger.info("Step 4/5: Saving outputs to %s", out_dir)

    parquet_path = out_dir / "static_features.parquet"
    features.to_parquet(parquet_path, index=False)
    logger.info("Feature table saved: %s (%d rows)", parquet_path, len(features))

    tif_path  = out_dir / "susceptibility.tif"
    gpkg_path = out_dir / "susceptibility.gpkg"

    export_susceptibility_raster(
        features, susc, tif_path,
        resolution=settings.grid_resolution_deg,
        crs_str=settings.target_crs,
        nodata=settings.nodata_value,
    )
    export_susceptibility_vector(features, susc, gpkg_path, crs_str=settings.target_crs)

    # ------------------------------------------------------------------ #
    # Step 5 — Write metadata
    # ------------------------------------------------------------------ #
    logger.info("Step 5/5: Writing provenance metadata.")
    meta = build_metadata(
        source=(
            "DEM: " + (settings.dem_path or "not configured") +
            " | Soil: " + (settings.soil_path or "not configured") +
            " | Geology: " + (settings.geology_path or "not configured") +
            " | LandCover: " + (settings.landcover_path or "not configured") +
            " | Drainage: " + (settings.drainage_path or "not configured") +
            " | Historical: " + (settings.historical_path or "not configured")
        ),
        crs=settings.target_crs,
        spatial_resolution=(
            f"{settings.grid_resolution_deg} degrees "
            f"(~{settings.grid_resolution_deg * 111_000:.0f} m at equator)"
        ),
        processing_version=settings.processing_version,
        nodata_handling=(
            "NaN used internally; never silently converted to 0. "
            f"GeoTIFF nodata sentinel = {settings.nodata_value}"
        ),
        geographic_coverage=f"bbox={bbox}",
        extra={
            "n_grid_cells": len(features),
            "susceptibility_weights": settings.susceptibility_weights,
            "susceptibility_weights_disclaimer": (
                "Weights are heuristic placeholders and are NOT scientifically validated."
            ),
            "columns": list(features.columns),
            "satellite_integration": (
                "Join satellite features on location_id (and optionally timestamp_utc). "
                "Pipeline does not fail if satellite data is absent."
            ),
            "temporal_assumption": (
                "historical_landslide uses ALL inventory records. "
                "Engineer 5 must filter to events pre-dating the prediction horizon."
            ),
        },
    )
    save_metadata(meta, out_dir / "metadata.json")

    logger.info("Pipeline complete.  Outputs: %s", out_dir)
    return parquet_path, tif_path, gpkg_path
