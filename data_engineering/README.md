# Data Engineering Module

**Engineer 1 — Data Engineering Foundation**
Branch: `engineer-1-data`

---

## Purpose

This module provides the reusable data ingestion, validation, preprocessing,
and spatial indexing pipeline for the landslide early-warning system.

It converts raw environmental and historical landslide datasets into a
standardised spatial-temporal dataset ready for consumption by:

- **Engineer 2** — GIS / susceptibility mapping
- **Engineer 4** — Rainfall / dynamic features
- **Engineer 5** — ML model training and inference

---

## Folder Structure

```
data_engineering/
├── __init__.py
├── ingestion/
│   ├── csv_loader.py         # Load tabular CSV files
│   ├── geojson_loader.py     # Load GeoJSON (reprojects to WGS-84)
│   └── shapefile_loader.py   # Load Shapefile / GeoPackage
├── preprocessing/
│   ├── landslide_standardizer.py  # Standardise raw records → contract schema
│   └── training_sampler.py        # Build training samples (positive + negative)
├── validation/
│   ├── coordinate_validator.py    # WGS-84 range checks
│   ├── timestamp_validator.py     # Multi-format → UTC ISO 8601
│   ├── duplicate_detector.py      # Flag (never drop) duplicate events
│   └── quality_pipeline.py        # Orchestrate all validation steps
├── spatial/
│   ├── grid.py               # Create spatial grid; assign grid_id to points
│   ├── location_id.py        # Deterministic location_id generator
│   └── spatial_ops.py        # Spatial join, clip, CRS transform utilities
├── schemas/
│   └── landslide_record.py   # LandslideRecord, ValidationReport, DataProvenance
├── pipelines/
│   └── build_dataset.py      # Main pipeline entry point
└── tests/
    ├── fixtures/
    │   ├── sample_landslides.csv     # Synthetic test data
    │   └── sample_landslides.geojson # Synthetic GeoJSON test data
    ├── test_coordinate_validator.py
    ├── test_timestamp_validator.py
    ├── test_duplicate_detector.py
    ├── test_spatial_grid.py
    ├── test_quality_pipeline.py
    └── test_pipeline_integration.py  # End-to-end integration test
```

---

## Supported Input Formats

| Format | Extension | Loader |
|--------|-----------|--------|
| CSV | `.csv` | `load_csv` |
| GeoJSON | `.geojson`, `.json` | `load_geojson` |
| Shapefile | `.shp` | `load_shapefile` |
| GeoPackage | `.gpkg` | `load_shapefile` |

All geospatial inputs are reprojected to **WGS-84 (EPSG:4326)** automatically.

---

## Standardised Output Schema

Every processed dataset contains these columns (data contract §2/§3/§4):

| Column | Type | Description |
|---|---|---|
| `location_id` | str | Deterministic spatial identifier |
| `latitude` | float | WGS-84 latitude (decimal degrees) |
| `longitude` | float | WGS-84 longitude (decimal degrees) |
| `geometry` | str | WKT point geometry |
| `timestamp_utc` | str | ISO 8601 UTC string (e.g. `2026-09-02T12:00:00Z`) |
| `historical_landslide` | int | `1` = event, `0` = non-event (training sampler) |
| `source` | str | Data source identifier |
| `grid_id` | str | Grid cell (present when `grid_config` is provided) |
| `is_duplicate` | bool | Duplicate flag — records are never silently dropped |

Static features (when present in raw data): `elevation`, `slope`, `aspect`, `soil_type`, `geology`, `land_cover`, `drainage_feature`.

---

## Example Usage

### Minimal — CSV to standardised DataFrame

```python
from data_engineering.pipelines import build_dataset

dataset, report, provenance = build_dataset(
    source_path="data/raw/landslides.csv",
    column_map={
        "latitude":  "lat",
        "longitude": "lon",
        "timestamp": "event_date",
    },
    source_name="GSI_landslide_inventory_2024",
)

print(report.to_dict())
# {
#   "records_total": 10000,
#   "records_valid": 9750,
#   "records_invalid": 250,
#   ...
# }
```

### With spatial grid assignment

```python
from data_engineering.pipelines import build_dataset
from data_engineering.spatial import GridConfig

grid = GridConfig(
    min_lat=26.0, max_lat=30.0,
    min_lon=85.0, max_lon=92.0,
    resolution_deg=0.01,     # configurable — ~1 km at equator
)

dataset, report, provenance = build_dataset(
    source_path="data/raw/landslides.csv",
    column_map={"latitude": "lat", "longitude": "lon", "timestamp": "date"},
    grid_config=grid,
    output_parquet="data/processed/landslide_events.parquet",
)
```

### Write to Parquet (for Engineer 5)

```python
dataset, _, _ = build_dataset(
    source_path="data/raw/landslides.csv",
    output_parquet="data/processed/events.parquet",
)
```

### GeoJSON input

```python
from data_engineering.pipelines import build_dataset

dataset, report, _ = build_dataset(
    source_path="data/raw/inventory.geojson",
    column_map={"timestamp": "date"},
)
```

### Training sample generation

```python
from data_engineering.pipelines import build_dataset
from data_engineering.preprocessing import TrainingSampler, SamplerConfig
from data_engineering.spatial import GridConfig, create_spatial_grid

# 1. Build the standardised positive events
pos_df, _, _ = build_dataset(source_path="data/raw/landslides.csv", ...)

# 2. Create the full spatial grid (shared with Engineer 2 / Engineer 5)
grid_config = GridConfig(26.0, 30.0, 85.0, 92.0, 0.01)
grid_gdf = create_spatial_grid(grid_config)

# 3. Sample
sampler = TrainingSampler(SamplerConfig(
    negative_to_positive_ratio=2.0,
    temporal_cutoff="2024-01-01T00:00:00Z",
    random_seed=42,
))
train_df = sampler.build(positive_df=pos_df, grid_df=grid_gdf)
```

### Use individual utilities

```python
from data_engineering.validation import validate_coordinates, normalize_timestamp
from data_engineering.spatial import generate_location_id

lat, lon = validate_coordinates(27.34, 88.61)
ts = normalize_timestamp("15/06/2022")       # → "2022-06-15T00:00:00Z"
loc_id = generate_location_id(lat, lon)      # → "LOC_+27.3400_+088.6100"
```

---

## Configuration

### Column Mapping

Raw datasets rarely use the contract column names.  Use `column_map`:

```python
column_map = {
    "latitude":  "Lat",          # raw column → contract name
    "longitude": "Lon",
    "timestamp": "EventDate",
}
```

### Grid Resolution

**Do not hard-code a resolution.**  Always pass a `GridConfig`:

```python
# For regional analysis (~10 km)
GridConfig(..., resolution_deg=0.1)

# For local analysis (~1 km)
GridConfig(..., resolution_deg=0.01)
```

The same `GridConfig` must be shared across engineers so that `grid_id`
and `location_id` are consistent.

---

## Missing Values

Per data contract §12:

> Missing values must **never** silently become zero.

- Missing coordinates → record excluded from output, counted in `ValidationReport`
- Missing timestamps → record excluded, counted in `ValidationReport`
- Missing features (elevation, etc.) → preserved as `None`/`NaN` in output

---

## Testing

```bash
# Run all tests (from project root)
source .venv/bin/activate
pytest data_engineering/tests/ -v

# Run just unit tests
pytest data_engineering/tests/ -v -k "not integration"

# Run with coverage
pytest data_engineering/tests/ --cov=data_engineering --cov-report=term-missing
```

---

## Integration Handoff

### Engineer 2 (GIS / Susceptibility)

Use `build_dataset()` to obtain the standardised `(location_id, latitude, longitude, geometry)` records.  The `grid_id` column connects to the same spatial grid.  Use `create_spatial_grid(config)` with the **shared** `GridConfig` to obtain the polygon layer for GIS operations.

### Engineer 4 (Rainfall / Dynamic Features)

Join dynamic rainfall features to the standardised dataset using `location_id` as the primary key and `timestamp_utc` for temporal alignment.

### Engineer 5 (ML Model)

```python
from data_engineering.pipelines import build_dataset

df, report, _ = build_dataset(
    source_path="data/processed/events.parquet",  # or CSV/GeoJSON
    column_map=...,
    grid_config=...,
)

# df columns match the data contract EnvironmentalFeatures schema:
# location_id, latitude, longitude, timestamp_utc, historical_landslide,
# elevation, slope, aspect, soil_type, geology, land_cover, ...
```

For training samples with negative events, use `TrainingSampler` (see example above).

---

## Data Provenance

Every call to `build_dataset()` returns a `DataProvenance` object:

```python
_, _, prov = build_dataset(...)
print(prov.to_dict())
# {
#   "source": "GSI_inventory",
#   "acquisition_datetime": "2026-09-01T00:00:00Z",
#   "geographic_coverage": "Sikkim, India",
#   "spatial_resolution": "point",
#   "preprocessing_version": "1.0.0",
#   ...
# }
```

---

## Known Limitations

1. **No real dataset bundled** — only synthetic fixtures are included.  Replace with real data paths in production.
2. **Negative sampling is spatial-grid-based** — requires `GridConfig` to cover the study area.  Small grids with few timestamps may produce fewer negatives than requested.
3. **GeoPackage multi-layer** — pass `layer=` kwarg via `load_shapefile` directly; not yet exposed through `build_dataset`.
4. **Large files** — for very large CSVs consider using `chunksize` in `load_csv` (via `read_kwargs`).  The current pipeline loads the entire file into memory.
5. **CRS assumption for GeoJSON** — files without CRS metadata are assumed to be EPSG:4326.
