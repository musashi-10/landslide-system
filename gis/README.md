# GIS Module — Static Landslide Susceptibility

## 1. Purpose

This module answers:

> **"Which areas are naturally more vulnerable to landslides?"**

It converts terrain and environmental data (DEM, soil, geology, land-cover,
drainage, and historical landslide inventory) into a standardised static
feature dataset that the ML engineer (Engineer 5) can consume directly.

---

## 2. Inputs

| Layer | Format | Required? | Notes |
|---|---|---|---|
| DEM (Digital Elevation Model) | GeoTIFF | Optional | Produces elevation, slope, aspect |
| Soil type map | Vector (GeoPackage / Shapefile) | Optional | Must have `soil_type` column |
| Geology map | Vector (GeoPackage / Shapefile) | Optional | Must have `geology` column |
| Land-cover map | Vector (GeoPackage / Shapefile) | Optional | Must have `land_cover` column |
| Drainage network | Vector (line / polygon) | Optional | Presence/absence per cell |
| Historical landslide inventory | Vector (point / polygon) | Optional | Converted to centroid if polygon |

All inputs are optional. The pipeline produces a valid (partial) output
even when no data sources are configured — with NaN for missing features.

---

## 3. Supported Formats

- **DEM**: GeoTIFF (any rasterio-readable raster)
- **Vectors**: GeoPackage (`.gpkg`), Shapefile (`.shp`), GeoJSON
- **Outputs**: Parquet, GeoTIFF, GeoPackage

---

## 4. Spatial Conventions

| Property | Value |
|---|---|
| Output CRS | `EPSG:4326` (WGS-84 geographic) — configurable |
| Resolution | `0.01°` default (~1.1 km at equator) — configurable |
| Grid IDs | `GRID_00001`, `GRID_00002`, … |
| Primary join key | `location_id` (same as `grid_id`) |
| Nodata | `NaN` internally; `-9999.0` in GeoTIFF; **never silently 0** |
| Timestamps | `UTC` / ISO-8601 where used |
| CRS mixing | **Forbidden** — all layers are validated/reprojected before joining |

---

## 5. Processing Pipeline

```text
 Input DEM
   └─► open_dem()          → validate CRS, reproject to EPSG:4326
   └─► compute_slope()     → Horn (1981) finite-difference, degrees
   └─► compute_aspect()    → clockwise from N, degrees (−1 for flat)

 Grid generation
   └─► generate_grid()     → configurable resolution, GRID_NNNNN IDs

 Feature extraction
   ├─► sample_raster_at_points()    ← elevation, slope, aspect
   ├─► spatial_join_to_grid()       ← soil_type, geology, land_cover
   ├─► point_to_grid_assignment()   ← historical_landslide (count)
   └─► drainage intersection        ← drainage_feature (0/1)

 Baseline susceptibility  [NOT the ML model]
   └─► compute_baseline_susceptibility()
         → min-max normalise numeric factors
         → encode categorical factors to proxy [0,1]
         → weighted sum → clip to [0,1]

 Outputs
   ├─► static_features.parquet    (Engineer 5 input)
   ├─► susceptibility.tif         (GeoTIFF)
   ├─► susceptibility.gpkg        (GeoPackage)
   └─► metadata.json              (provenance)
```

---

## 6. Outputs

### A. `static_features.parquet`

| Column | Type | Description |
|---|---|---|
| `location_id` | str | Primary join key (e.g. `GRID_00001`) |
| `latitude` | float | Centroid latitude (degrees, WGS-84) |
| `longitude` | float | Centroid longitude (degrees, WGS-84) |
| `geometry` | str | WKT polygon of the grid cell |
| `elevation` | float\|NaN | Metres above sea level |
| `slope` | float\|NaN | Degrees (0=flat, 90=vertical) |
| `aspect` | float\|NaN | Degrees CW from North (−1=flat) |
| `soil_type` | str\|None | Soil classification |
| `geology` | str\|None | Geological unit |
| `land_cover` | str\|None | Land-cover class |
| `drainage_feature` | int | 1 = drainage feature present, 0 = absent |
| `historical_landslide` | int | Count of historical landslide events in cell |
| `susceptibility` | float\|NaN | Baseline susceptibility [0–1] (**not validated**) |

### B. `susceptibility.tif`

GeoTIFF raster, one band, float32, EPSG:4326. Nodata = −9999.0.

### C. `susceptibility.gpkg`

GeoPackage polygon layer for visual inspection in QGIS/ArcGIS.

### D. `metadata.json`

Provenance sidecar: source paths, CRS, resolution, processing version,
timestamp, nodata handling, weights disclaimer.

---

## 7. Example Usage

```python
from gis import build_static_feature_dataset
from gis.config import GISSettings

settings = GISSettings(
    dem_path="data/dem.tif",
    soil_path="data/soil.gpkg",
    geology_path="data/geology.gpkg",
    landcover_path="data/landcover.gpkg",
    drainage_path="data/drainage.gpkg",
    historical_path="data/historical_landslides.gpkg",
    grid_resolution_deg=0.01,   # ~1.1 km
    output_dir="gis_outputs",
)

# Eastern Sikkim, India (known landslide-prone region)
parquet_path, tif_path, gpkg_path = build_static_feature_dataset(
    bbox=(88.0, 27.0, 89.0, 28.0),
    settings=settings,
)

import pandas as pd
features = pd.read_parquet(parquet_path)
print(features.head())
```

**Minimal run (no real data):**

```python
from gis import build_static_feature_dataset

# All layers optional — produces NaN features but valid schema
parquet_path, tif_path, gpkg_path = build_static_feature_dataset(
    bbox=(88.0, 27.0, 88.2, 27.2),
    output_dir="gis_outputs",
)
```

---

## 8. Configuration

All settings are in [`gis/config.py`](gis/config.py) and can be overridden
via environment variables (prefix `GIS_`) or a `.env` file.

| Variable | Default | Description |
|---|---|---|
| `GIS_TARGET_CRS` | `EPSG:4326` | Output coordinate reference system |
| `GIS_GRID_RESOLUTION_DEG` | `0.01` | Grid cell size in degrees |
| `GIS_DEM_PATH` | `""` | Path to DEM GeoTIFF |
| `GIS_SOIL_PATH` | `""` | Path to soil vector |
| `GIS_GEOLOGY_PATH` | `""` | Path to geology vector |
| `GIS_LANDCOVER_PATH` | `""` | Path to land-cover vector |
| `GIS_DRAINAGE_PATH` | `""` | Path to drainage vector |
| `GIS_HISTORICAL_PATH` | `""` | Path to historical landslide vector |
| `GIS_OUTPUT_DIR` | `gis_outputs` | Output directory |
| `GIS_NODATA_VALUE` | `-9999.0` | GeoTIFF nodata sentinel |
| `GIS_PROCESSING_VERSION` | `0.1.0` | Pipeline version for provenance |

**Susceptibility weights** (in `GISSettings.susceptibility_weights`):

> ⚠️ **These weights are NOT scientifically validated.**  
> They are heuristic placeholders for development and integration testing.  
> Engineer 5's ML model supersedes them.

---

## 9. Tests

```bash
# Run all tests (integration + GIS)
pytest -q

# Run GIS tests only
pytest gis/tests/ -v
```

Tests cover:

- CRS transformation (raster + vector)
- Elevation extraction (in-bounds, out-of-bounds, nodata)
- Slope calculation (range, shape, NaN, flat, steep)
- Aspect calculation (range, shape, NaN, flat, north-facing)
- Raster alignment (same-grid, coarser target, cross-CRS)
- Grid generation (cell count, uniqueness, centroids, errors)
- Spatial joins (soil, geology, point-to-grid, CRS mismatch)
- Missing raster values (NaN propagation, never silently 0)
- Invalid geometry (degenerate bbox, out-of-bounds points)
- Output schema (all required columns, data-contract compliance)
- Baseline susceptibility (range, NaN handling, ordering)
- Full pipeline with synthetic data (no external downloads)
- Minimal pipeline (no data sources configured)

---

## 10. Integration Instructions for Engineer 5 (ML)

**Read the Parquet file:**

```python
import pandas as pd

features = pd.read_parquet("gis_outputs/static_features.parquet")
# features has columns: location_id, latitude, longitude, geometry,
# elevation, slope, aspect, soil_type, geology, land_cover,
# drainage_feature, historical_landslide, susceptibility
```

**Join with satellite features (Engineer 3):**

```python
satellite = pd.read_parquet("satellite_outputs/features.parquet")
combined = features.merge(satellite, on="location_id", how="left")
# Pipeline never fails if satellite features are absent
```

**Join with rainfall features (Engineer 1):**

```python
rainfall = pd.read_parquet("rainfall_outputs/features.parquet")
combined = features.merge(rainfall, on=["location_id", "timestamp_utc"], how="left")
```

**Validate against shared schema:**

```python
from shared.schemas import EnvironmentalFeatures

for _, row in features.iterrows():
    record = EnvironmentalFeatures(
        location_id=row["location_id"],
        latitude=row["latitude"],
        longitude=row["longitude"],
        elevation=row.get("elevation"),
        slope=row.get("slope"),
        # ...
    )
```

**Temporal assumption — IMPORTANT:**

The `historical_landslide` column uses **all** inventory records.
When performing train/test splits, Engineer 5 must filter the inventory
to events that pre-date the prediction horizon to avoid data leakage.

---

## 11. Known Limitations

1. **Baseline susceptibility weights are not scientifically validated.**  
   They must be replaced by Engineer 5's ML model for operational use.

2. **Geographic CRS resolution** (degrees) introduces subtle distortion at
   high latitudes. For polar regions, use a projected CRS.

3. **Raster-to-point sampling** uses nearest-neighbour pixel lookup. For
   very coarse rasters relative to the grid, consider bilinear interpolation.

4. **Categorical encodings** (land-cover, soil, geology proxies) only cover
   common classes. Unknown categories default to 0.5 (neutral).

5. **Large DEMs** are loaded fully into memory. For production-scale DEMs,
   implement windowed reading (rasterio `Window`).

6. **Historical landslide count** is a raw count, not a density or binary
   indicator. Engineer 5 should decide the appropriate encoding.

7. **Satellite features** (Engineer 3) are not included in this module.
   They must be joined on `location_id` externally.
