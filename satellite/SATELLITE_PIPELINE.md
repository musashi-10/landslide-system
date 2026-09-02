# Satellite Feature Pipeline

**Engineer 3 — Remote Sensing Module**
Branch: `engineer-3-satellite`

---

## Overview

This module extracts standardized, spatially-referenced features from
satellite imagery for use in the landslide susceptibility model.

The pipeline output is a feature table (Parquet) with one row per
spatial location, joinable with Engineer 2's GIS static features and
Engineer 5's ML feature table via `location_id`.

> ⚠ **Scientific note**: Satellite-derived features are *inputs* to the
> susceptibility model. This pipeline does NOT directly predict
> landslides. It provides environmental indicators that, combined with
> terrain, soil, geology, and rainfall data, improve the AI risk model.

---

## Data Sources

### Sentinel-2 MSI L2A (Primary)

| Property | Value |
|---|---|
| Provider | ESA Copernicus / AWS Open Data |
| Product | Sentinel-2 MSI Level-2A (surface reflectance) |
| Bands used | B04 (Red, 10 m), B08 (NIR, 10 m), B11 (SWIR, 20 m) |
| Spatial resolution | 10 m (native for B04/B08) |
| Temporal resolution | ~5 days revisit |
| Availability | Global, 2015–present |
| License | Free / open access |
| Access | [Copernicus Open Access Hub](https://scihub.copernicus.eu/) / [AWS](https://registry.opendata.aws/sentinel-2/) |

**Why Sentinel-2?**
- Free and publicly available
- 10 m resolution is appropriate for slope-scale landslide features
- 5-day revisit enables temporal change detection
- L2A products are atmospherically corrected (surface reflectance)

### Sentinel-1 SAR (Future — documented, not implemented)

| Property | Value |
|---|---|
| Why relevant | C-band SAR penetrates cloud cover (critical for monsoon seasons) |
| Bands | VV, VH polarizations |
| Resolution | 10 m (IW mode) |
| Status | **NOT implemented in this prototype** |
| Limitation | Requires coherent pairs for InSAR; additional preprocessing needed |

---

## Features Produced

| Field | Formula | Range | Why |
|---|---|---|---|
| `ndvi` | (NIR − Red) / (NIR + Red) | [−1, 1] | Vegetation density; sparse veg → higher landslide susceptibility |
| `bare_surface_index` | (SWIR + Red − NIR) / (SWIR + Red + NIR) | [−1, 1] | Exposed ground; bare slopes are more erosion-prone |
| `ndvi_change` | NDVI(t2) − NDVI(t1) | (−2, 2) | Vegetation loss = possible disturbance; negative = concerning |
| `land_cover` | Rule-based threshold on NDVI+BSI | categorical | Standard vulnerability feature; required by data contract |

### Land Cover Categories
`forest` · `shrub_grass` · `bare` · `water` · `unknown`

> ⚠ Land cover classification is rule-based (not a trained classifier).
> For production, replace with ESA WorldCover 10 m or Google Dynamic World.

---

## Pipeline Stages

```
Stage A  Ingestion      satellite/ingestion/loader.py
Stage B  Preprocessing  satellite/preprocessing/pipeline.py
Stage C  Features       satellite/features/indices.py
Stage D  Alignment      satellite/spatial/alignment.py
Stage E  Change detect  satellite/change_detection/detector.py
Stage F  Vision model   NOT IMPLEMENTED — see Limitations
```

---

## Quickstart

### 1. In-memory (for testing / integration)

```python
import numpy as np
from satellite import extract_features_from_arrays, SatelliteConfig
from satellite.tests.conftest import SimpleTransform

# Create synthetic 100×100 scene
rng = np.random.default_rng(42)
red  = rng.uniform(0.05, 0.25, (100, 100)).astype("float32")
nir  = rng.uniform(0.20, 0.50, (100, 100)).astype("float32")
swir = rng.uniform(0.05, 0.30, (100, 100)).astype("float32")

transform = SimpleTransform(
    pixel_width=0.0001, pixel_height=0.0001,
    origin_x=88.0, origin_y=27.01,
)

config = SatelliteConfig(processing_version="v1")

records = extract_features_from_arrays(
    band_arrays={"red": red, "nir": nir, "swir": swir},
    transform=transform,
    crs="EPSG:4326",
    acquisition_time="2024-06-15T10:23:00Z",
    config=config,
)

# Export to Parquet
import pandas as pd
df = pd.DataFrame([r.model_dump() for r in records])
df.to_parquet("satellite_features.parquet", index=False)
print(df[["location_id", "ndvi", "bare_surface_index", "land_cover"]].head())
```

### 2. From a real Sentinel-2 scene directory

```python
from satellite import extract_satellite_features, SatelliteConfig

config = SatelliteConfig(
    bbox=(88.0, 27.0, 89.0, 28.0),
    processing_version="v1",
)

records = extract_satellite_features(
    scene_path="/data/sentinel2/S2A_MSIL2A_20240615T043701_N0510_R090_T45RVK/",
    config=config,
    acquisition_time="2024-06-15T04:37:01Z",
)
```

### 3. With change detection (two dates)

```python
records = extract_features_from_arrays(
    band_arrays={"red": red_t2, "nir": nir_t2},
    transform=transform,
    crs="EPSG:4326",
    acquisition_time="2024-07-20T10:00:00Z",
    ndvi_t1=ndvi_from_t1,     # NumPy array from earlier date
)
# Records will include ndvi_change column
```

---

## Integration with Other Engineers

### Engineer 2 (GIS)

```python
from satellite.spatial.alignment import records_to_geodataframe

gdf = records_to_geodataframe([r.model_dump() for r in records])
# gdf is CRS EPSG:4326, geometry=POINT
# Join on location_id:
merged = gis_gdf.merge(gdf[["location_id", "ndvi", "bare_surface_index"]], on="location_id")
```

### Engineer 5 (ML)

```python
import pandas as pd

df = pd.DataFrame([r.model_dump() for r in records])
# df has columns: location_id, ndvi, bare_surface_index, ndvi_change, land_cover, ...
# Join on location_id with other feature tables
ml_features = static_df.merge(df, on="location_id", how="left")
```

---

## Output Schema (`SatelliteFeatureRecord`)

```
location_id          str     PRIMARY KEY — join key
latitude             float   Decimal degrees (EPSG:4326)
longitude            float   Decimal degrees (EPSG:4326)
geometry             str     WKT POINT (optional)
acquisition_time     str     ISO 8601 UTC
ndvi                 float?  [-1, 1]
bare_surface_index   float?  [-1, 1]
ndvi_change          float?  (t2-t1 NDVI delta)
land_cover           str?    forest/shrub_grass/bare/water/unknown
source               str     "Sentinel-2"
spatial_resolution_m int     10
processing_version   str     "v1"
source_crs           str?    e.g. "EPSG:32645"
output_crs           str     "EPSG:4326" (always)
```

> **Missing values**: All optional features are `None` when unavailable.
> Never `0.0` (per data contract Section 12).

---

## CRS Documentation

| Stage | CRS |
|---|---|
| Source raster | EPSG:32645 (UTM zone 45N, typical for Himalayas) or native |
| Internal processing | Source CRS (no intermediate reprojection) |
| Pixel centroid calculation | Source CRS |
| Reprojection | pyproj.Transformer (always_xy=True) |
| Output coordinates | EPSG:4326 (WGS-84) |
| GeoDataFrame | EPSG:4326 |

---

## Configuration (`.env`)

```bash
SATELLITE_CACHE_DIR=.satellite_cache        # Where to cache downloads
SATELLITE_PROCESSING_VERSION=v1            # Version tag on outputs
SATELLITE_SPATIAL_RESOLUTION_M=10         # Output pixel resolution
SATELLITE_CLOUD_THRESHOLD=0.30            # Max cloud fraction
```

---

## Running Tests

```bash
# Full suite (including existing contract tests)
pytest -q

# Satellite tests only
pytest satellite/tests/ -v

# Specific test file
pytest satellite/tests/test_indices.py -v
```

---

## Actual Data Availability Assessment

This is critical per the project specification:

| Resource | Count / Amount |
|---|---|
| Sentinel-2 scenes (global archive) | Millions (2015–present) |
| Sentinel-2 scenes for Nepal/Himalayas | Thousands per year |
| **Labeled landslide satellite examples** | **0 in this repository** |
| Labeled spatial training samples | 0 in this repository |
| Usable image pairs for change detection | 0 in this repository |

**Consequence**: No supervised vision model can be trained on in-repository data.
The prototype uses index-based features (NDVI, BSI) which require **no labels**.

For labeled training data, consider:
- [NASA Landslide Viewer](https://landslides.nasa.gov/)
- [Cooperative Open Online Landslide Repository (COOLR)](https://gpm.nasa.gov/landslides/projects/coolr.html)
- [BGS National Landslide Database](https://www.bgs.ac.uk/datasets/national-landslide-database/)

---

## Limitations

1. **No real satellite data in tests** — synthetic NumPy arrays only.
   Real satellite files are hundreds of MB; they must be downloaded separately.

2. **No labeled landslide satellite dataset** — no supervised model trained.
   Stage F (pretrained vision model) is NOT implemented and is not justified
   at this stage of the project.

3. **Land cover is rule-based** — threshold values are approximate.
   For production, use ESA WorldCover 10 m or Google Dynamic World.

4. **Cloud masking is simplified** — full L2A cloud masking requires the
   Sentinel-2 SCL (Scene Classification Layer). The prototype accepts an
   externally supplied boolean mask.

5. **Change detection requires co-registered images** — temporal pairs
   must cover the same spatial extent and be on the same pixel grid.
   This is not validated on real data.

6. **Sentinel-1 SAR not implemented** — cloud-penetrating SAR imagery
   would improve change detection during monsoon season.

7. **No temporal aggregation** — the pipeline processes one image pair.
   Multi-temporal compositing (e.g., median over a season) is not implemented.

---

## File Tree

```
satellite/
├── __init__.py                    Public API
├── config.py                      SatelliteConfig
├── pipeline.py                    extract_satellite_features()
├── SATELLITE_PIPELINE.md          This file
├── ingestion/
│   ├── __init__.py
│   └── loader.py                  load_scene(), scene_from_arrays()
├── preprocessing/
│   ├── __init__.py
│   └── pipeline.py                mask_invalid(), normalize_band(), preprocess_scene()
├── features/
│   ├── __init__.py
│   └── indices.py                 compute_ndvi(), compute_bsi(), classify_land_cover()
├── change_detection/
│   ├── __init__.py
│   └── detector.py                compute_ndvi_change(), detect_disturbance()
├── spatial/
│   ├── __init__.py
│   └── alignment.py               raster_to_points(), make_location_id(), records_to_geodataframe()
├── schemas/
│   ├── __init__.py
│   └── satellite_schema.py        SatelliteFeatureRecord (Pydantic)
└── tests/
    ├── __init__.py
    ├── conftest.py                 Synthetic fixtures
    ├── fixtures/README.md          Fixture convention
    ├── test_preprocessing.py
    ├── test_indices.py
    ├── test_change_detection.py
    ├── test_alignment.py
    ├── test_schema.py
    └── test_pipeline.py
```
