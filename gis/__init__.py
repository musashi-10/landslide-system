"""
GIS module for static landslide susceptibility.

Provides:
  - DEM ingestion and terrain derivative calculation (elevation, slope, aspect)
  - Spatial grid generation
  - Multi-layer feature extraction (soil, geology, land-cover, drainage, historical)
  - Spatial alignment utilities (CRS, reprojection, raster alignment)
  - Baseline susceptibility computation
  - Clean public API: build_static_feature_dataset()

Engineer 5 (ML) should call build_static_feature_dataset() and read the
resulting Parquet file.  The geometry / satellite columns are optional and
joined via location_id.
"""

try:
    from gis.pipeline import build_static_feature_dataset
    __all__ = ["build_static_feature_dataset"]
except ImportError:
    # rasterio / geopandas not installed in this environment.
    # GIS functionality is unavailable but the package can still be imported.
    __all__ = []
