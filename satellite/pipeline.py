"""
satellite/pipeline.py
-----------------------
Public entry point for the satellite feature pipeline.

The two public functions are:

``extract_satellite_features(scene_path, config, acquisition_time)``
    Full pipeline from a directory of Sentinel-2 GeoTIFFs to a list of
    ``SatelliteFeatureRecord``.  Requires rasterio, pyproj, geopandas.

``extract_features_from_arrays(band_arrays, transform, crs, ...)``
    In-memory version for unit tests and downstream integration — no file
    I/O required.  Accepts plain NumPy arrays.

Engineer 5 integration
-----------------------
The output of ``extract_satellite_features`` can be converted to a
Pandas DataFrame and serialized to Parquet:

    records = extract_satellite_features(scene_dir, config, acq_time)
    import pandas as pd
    df = pd.DataFrame([r.model_dump() for r in records])
    df.to_parquet("satellite_features.parquet", index=False)

The ``location_id`` column is the join key to all other feature tables.

Engineer 2 integration
-----------------------
Use ``records_to_geodataframe`` from ``satellite.spatial.alignment`` to
produce a GeoDataFrame (EPSG:4326) that can be spatially joined with
Engineer 2's GIS outputs.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .config import SatelliteConfig
from .ingestion.loader import load_scene, scene_from_arrays
from .preprocessing.pipeline import preprocess_scene
from .features.indices import compute_ndvi, compute_bsi, classify_land_cover
from .change_detection.detector import compute_ndvi_change, detect_disturbance
from .spatial.alignment import raster_to_points, assign_location_id
from .schemas.satellite_schema import SatelliteFeatureRecord

logger = logging.getLogger(__name__)


def extract_features_from_arrays(
    band_arrays: dict[str, np.ndarray],
    transform: Any,
    crs: str,
    acquisition_time: str,
    config: SatelliteConfig | None = None,
    ndvi_t1: np.ndarray | None = None,
    cloud_masks: dict[str, np.ndarray] | None = None,
) -> list[SatelliteFeatureRecord]:
    """Extract satellite features from in-memory NumPy arrays.

    This is the primary entry point for unit tests and for pipeline
    components that already have arrays in memory.

    Parameters
    ----------
    band_arrays : dict
        Must contain at minimum ``"red"`` and ``"nir"`` keys.
        Optionally ``"swir"`` for BSI calculation.
        Values must be float arrays in [0, 1] (surface reflectance).
    transform : affine.Affine or compatible
        Affine geotransform in the given CRS.
    crs : str
        CRS of the input arrays (e.g. ``"EPSG:4326"`` or ``"EPSG:32645"``).
    acquisition_time : str
        ISO 8601 UTC acquisition timestamp of this scene.
    config : SatelliteConfig, optional
        Pipeline configuration.  Uses defaults if None.
    ndvi_t1 : np.ndarray, optional
        NDVI from a prior acquisition (t1) for change detection.
        If None, ``ndvi_change`` is not computed.
    cloud_masks : dict, optional
        Per-band boolean cloud masks.  True = cloud pixel.

    Returns
    -------
    list[SatelliteFeatureRecord]
        One record per valid (unmasked) pixel.
    """
    config = config or SatelliteConfig()

    # --- Ingestion: wrap arrays in scene dict ---
    scene = scene_from_arrays(band_arrays, transform, crs)

    # --- Preprocessing ---
    scene = preprocess_scene(scene, cloud_masks=cloud_masks)
    bands = scene["bands"]

    red = bands.get("red")
    nir = bands.get("nir")
    swir = bands.get("swir")

    if red is None or nir is None:
        raise ValueError(
            "'red' and 'nir' bands are required for feature extraction."
        )

    # --- Feature extraction ---
    ndvi = compute_ndvi(red, nir)
    bsi = compute_bsi(red, nir, swir) if swir is not None else None
    land_cover_arr = classify_land_cover(ndvi, bsi if bsi is not None else ndvi)

    # --- Change detection (optional) ---
    ndvi_change_arr: np.ma.MaskedArray | None = None
    if ndvi_t1 is not None:
        ndvi_t1_ma = np.ma.MaskedArray(ndvi_t1.astype(np.float32))
        ndvi_change_arr = compute_ndvi_change(ndvi_t1_ma, ndvi)
        result = detect_disturbance(ndvi_change_arr, config.ndvi_change_threshold)
        logger.info(
            "Change detection: %d pixels flagged as disturbance (threshold=%.2f)",
            result["flagged_pixel_count"], result["threshold_used"],
        )

    # --- Build feature arrays dict for spatial alignment ---
    feature_arrays: dict[str, np.ma.MaskedArray] = {"ndvi": ndvi}
    if bsi is not None:
        feature_arrays["bare_surface_index"] = bsi
    if ndvi_change_arr is not None:
        feature_arrays["ndvi_change"] = ndvi_change_arr

    # --- Spatial alignment → point records ---
    raw_records = raster_to_points(
        feature_arrays=feature_arrays,
        transform=transform,
        source_crs=crs,
        acquisition_time=acquisition_time,
        source=config.source,
        spatial_resolution_m=config.spatial_resolution_m,
        processing_version=config.processing_version,
    )

    # Attach land_cover (string array — handle separately)
    land_cover_flat = land_cover_arr.ravel()
    # land_cover values are indexed by the same pixel order as the feature arrays
    # re-derive the valid pixel indices from the first feature
    ndvi_flat_mask = np.ma.getmaskarray(ndvi).ravel()
    valid_indices = np.where(~ndvi_flat_mask)[0]

    for i, rec in enumerate(raw_records):
        if i < len(valid_indices):
            lc = str(land_cover_flat[valid_indices[i]])
            rec["land_cover"] = lc if lc != "unknown" else None

    assign_location_id(raw_records)

    # --- Validate through schema ---
    records: list[SatelliteFeatureRecord] = []
    for rec in raw_records:
        try:
            records.append(SatelliteFeatureRecord(**rec))
        except Exception as exc:
            logger.warning("Skipping invalid record: %s  error=%s", rec.get("location_id"), exc)

    logger.info(
        "extract_features_from_arrays: produced %d validated records", len(records)
    )
    return records


def extract_satellite_features(
    scene_path: str,
    config: SatelliteConfig | None = None,
    acquisition_time: str = "1970-01-01T00:00:00Z",
    ndvi_t1_path: str | None = None,
) -> list[SatelliteFeatureRecord]:
    """Full pipeline: Sentinel-2 scene directory → feature records.

    Requires rasterio, pyproj, geopandas.  For unit tests, prefer
    :func:`extract_features_from_arrays`.

    Parameters
    ----------
    scene_path : str
        Path to a directory containing per-band Sentinel-2 GeoTIFFs.
        Expected band file patterns: ``*B04*.tif``, ``*B08*.tif``, ``*B11*.tif``.
    config : SatelliteConfig, optional
        Pipeline configuration.  Defaults to ``SatelliteConfig()``.
    acquisition_time : str
        ISO 8601 UTC timestamp for this scene.
    ndvi_t1_path : str, optional
        Path to a single-band NDVI GeoTIFF from a prior acquisition (t1)
        for change detection.  If None, change detection is skipped.

    Returns
    -------
    list[SatelliteFeatureRecord]
    """
    from .ingestion.loader import load_raster_band  # noqa: PLC0415

    config = config or SatelliteConfig()

    scene = load_scene(scene_path)
    bands = scene["bands"]

    ndvi_t1: np.ndarray | None = None
    if ndvi_t1_path:
        ndvi_t1_arr, _, _ = load_raster_band(ndvi_t1_path, band_index=1)
        ndvi_t1 = ndvi_t1_arr.astype(np.float32)

    return extract_features_from_arrays(
        band_arrays={k: np.ma.filled(v, 0.0) for k, v in bands.items()},
        transform=scene["transform"],
        crs=scene["crs"],
        acquisition_time=acquisition_time,
        config=config,
        ndvi_t1=ndvi_t1,
    )
