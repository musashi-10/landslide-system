# Environment Pipeline — Dynamic Trigger Feature Documentation

**Engineer 4 | Branch: `engineer-4-environment`**

---

## 1. Purpose

This module converts raw weather/rainfall observations and rainfall history into
standardised dynamic environmental features that feed into the landslide risk
model (Engineer 5).

```
Rainfall / Weather Data
        ↓
WeatherProvider (Open-Meteo or Mock)
        ↓
RainfallIngestor (validation + normalisation)
        ↓
Time-series feature computation
        ↓
DynamicRecord (standardised output)
        ↓
Engineer 5 ML Model   /   Engineer 6 Dashboard API
```

The pipeline does **not** predict landslides. It provides environmental
trigger conditions that the ML model uses together with static GIS features.

---

## 2. Data Sources

### 2.1 Open-Meteo (production provider)

| Property | Value |
|---|---|
| Source | [Open-Meteo](https://open-meteo.com) (ERA5-Land reanalysis + ICON NWP forecast) |
| API key required | No (free tier) |
| Spatial resolution | ~1 km (ERA5-Land) / ~2–13 km (ICON forecast) |
| Temporal resolution | 1-hour accumulated precipitation |
| Update frequency | ERA5-Land: ~5-day reanalysis lag; Forecast (ICON): every 6 hours |
| Units | mm (millimetres per hour, accumulated) |
| License | CC BY 4.0 |
| Rainfall value type | `accumulated_per_hour` |

**Limitations:**
- ERA5-Land has a ~5-day lag from real-time. Very recent hours fall back to forecast fields.
- Grid-cell-average values may differ from point rain-gauge readings.
- Sub-kilometre spatial variability is not captured.
- Subject to API rate limits (free tier).

### 2.2 Mock Provider (testing / offline development)

Deterministic synthetic data generator. Reads from
`environment/tests/fixtures/synthetic_rainfall.csv` (8-day hourly series for
two locations), falling back to a sinusoidal generator if the file is absent.
No external API calls.

---

## 3. Provider Abstraction

All providers implement `WeatherProvider` (abstract base class):

```python
class WeatherProvider(ABC):
    def get_observations(latitude, longitude, start_utc, end_utc) -> List[RawObservation]
    def get_forecast(latitude, longitude, horizon_hours) -> List[ForecastRecord]
    def provenance -> DataProvenance
```

The factory function selects a provider by name:

```python
from environment.providers import get_provider
provider = get_provider("open_meteo")   # production
provider = get_provider("mock")          # testing
```

API keys come from `.env` (via `shared/config/settings.py`). Never hard-coded.

---

## 4. Rainfall Features

All rainfall values are in **millimetres (mm)**.

### 4.1 Required features (data contract §6)

| Feature | Description | Value type | Window |
|---|---|---|---|
| `rainfall_1h` | 1-hour accumulated rainfall | accumulation (mm) | 1 h |
| `rainfall_6h` | 6-hour rolling accumulated rainfall | accumulation (mm) | 6 h |
| `rainfall_24h` | 24-hour rolling accumulated rainfall | accumulation (mm) | 24 h |
| `rainfall_3d` | 3-day (72-hour) rolling accumulated rainfall | accumulation (mm) | 72 h |
| `rainfall_7d` | 7-day (168-hour) rolling accumulated rainfall | accumulation (mm) | 168 h |
| `forecast_rainfall` | Forecast accumulation over configured horizon | accumulation (mm) | 24 h (configurable) |
| `moisture_indicator` | Estimated environmental moisture proxy [0, 1] | dimensionless | derived |

### 4.2 Additional features (optional, computed when history is sufficient)

| Feature | Description | Value type |
|---|---|---|
| `rainfall_intensity` | Instantaneous rainfall rate (last 1 hour) | rate (mm/hr) |
| `antecedent_rainfall` | Exponentially-weighted antecedent index | index (mm-equivalent) |
| `rainfall_anomaly` | Deviation of rainfall_24h from 7-day mean | mm (can be negative) |

### 4.3 Window configuration

All window sizes are centralised in `environment/config.py`:

```python
EnvironmentConfig(
    window_hours={"1h": 1, "6h": 6, "24h": 24, "3d": 72, "7d": 168},
    forecast_horizon_hours=24,
    moisture_decay_factor=0.85,
)
```

No window sizes are hard-coded elsewhere.

---

## 5. Moisture Indicator

**IMPORTANT**: This is an **estimated proxy**, not a physical soil-moisture measurement.

- **Source**: Derived from recent rainfall history (Open-Meteo ERA5-Land or mock)
- **Method**: Sigmoid-normalised antecedent rainfall index
- **Formula**: `moisture_indicator = sigmoid((API / saturation_scale) * 2 - 2)`
  where `API` = exponentially-weighted antecedent rainfall
- **Spatial resolution**: Same as rainfall source (~1 km ERA5-Land)
- **Temporal resolution**: Computed hourly (pipeline run frequency)
- **Update frequency**: Updated every pipeline run
- **Range**: [0, 1]. 0 = very dry antecedent; 1 = saturated (asymptotic)
- **Limitations**:
  - Does not model evapotranspiration, drainage, or runoff
  - Cannot detect soil dryness from vegetation uptake alone
  - For production: replace with SMAP / Copernicus CGLS soil moisture product

---

## 6. Timestamp Convention

- All timestamps: `timestamp_utc` in ISO 8601 UTC format
- Example: `2026-09-02T12:00:00Z`
- Timezone-naive inputs are treated as UTC
- Conversion from any timezone to UTC is handled by `preprocessing.timestamp_utils`
- Rainfall windows are computed looking **backwards** from `timestamp_utc`

---

## 7. Spatial Mapping

Provider grid coordinates → project `location_id` mapping:

```
Provider API grid point (nearest to requested lat/lon)
        ↓
SpatialMapper (nearest-neighbour, Haversine distance)
        ↓
project location_id  (LOC_{lat:+.4f}_{lon:+08.4f})
```

- **Method**: Nearest-neighbour (Haversine distance)
- **Fallback**: If no project location within `max_distance_km` (default 5 km),
  `location_id` is generated directly from requested coordinates
- **format**: Compatible with Engineer 1's `generate_location_id`
  (`LOC_{lat:+.4f}_{lon:+08.4f}`)

---

## 8. Missing Data Strategy

| Situation | Behaviour |
|---|---|
| Missing observation (NaN in time series) | Excluded from rolling sum; other hours contribute |
| All observations in window are NaN | Feature returns `None` |
| Empty observation DataFrame | All features return `None` |
| Provider timeout/unavailable | Error logged; all rainfall fields return `None` |
| Forecast unavailable | `forecast_rainfall = None` (never silently → 0) |
| Partial forecast (some hours None) | Partial sum returned with warning log |

**Critical rule**: Missing values are **never** silently converted to 0.
`None` (Python) / `NaN` (DataFrame) is always used to represent absence of data.

---

## 9. Caching

- **Type**: In-memory TTL cache (`LocalCache`)
- **TTL**: 3600 seconds (1 hour) — configurable
- **Thread-safe**: Yes (threading.Lock)
- **Scope**: Three cache levels:
  1. Observation DataFrame per (lat, lon, start, end, provider)
  2. Forecast list per (lat, lon, horizon, provider)
  3. Computed DynamicRecord per (location_id, timestamp_utc)
- **Disable for testing**: Set `cache_ttl_seconds=0`

---

## 10. Public Interface

### For Engineer 5 (ML model)

```python
from environment import build_dynamic_features, get_dynamic_conditions
from environment.providers import get_provider

provider = get_provider("open_meteo")  # or "mock" for testing

# Single location
record = build_dynamic_features(
    location_id="LOC_+27.1230_+88.4560",
    latitude=27.123,
    longitude=88.456,
    timestamp_utc="2026-09-02T12:00:00Z",
    provider=provider,
)

# record.rainfall_1h, .rainfall_6h, .rainfall_24h, etc.

# Convert to flat dict (JSON-serialisable)
from environment import dynamic_record_to_dict
d = dynamic_record_to_dict(record)
```

The `DynamicRecord` is directly mergeable into `shared.schemas.EnvironmentalFeatures`:

```python
from shared.schemas import EnvironmentalFeatures

ef = EnvironmentalFeatures(
    location_id=record.location_id,
    latitude=record.latitude,
    longitude=record.longitude,
    timestamp_utc=record.timestamp_utc,
    # static fields from Engineer 2 ...
    rainfall_1h=record.rainfall_1h,
    rainfall_6h=record.rainfall_6h,
    rainfall_24h=record.rainfall_24h,
    rainfall_3d=record.rainfall_3d,
    rainfall_7d=record.rainfall_7d,
    forecast_rainfall=record.forecast_rainfall,
    moisture_indicator=record.moisture_indicator,
)
```

### Batch processing

```python
from environment import DynamicFeaturePipeline, LocationSpec

pipeline = DynamicFeaturePipeline(provider=provider)
result = pipeline.run(locations=[
    LocationSpec("LOC_+27.1230_+88.4560", 27.123, 88.456),
    LocationSpec("LOC_+27.5000_+88.7500", 27.5, 88.75),
], timestamp_utc="2026-09-02T12:00:00Z")

df = result.to_dataframe()  # pandas DataFrame for ML consumption
```

### For Engineer 6 (backend / dashboard API)

Mount the FastAPI router:

```python
# In backend/main.py
from environment.APIs import router as env_router
app.include_router(env_router, prefix="/environment", tags=["environment"])
```

Endpoints:

```
GET /environment/health
    → {"status": "ok", "service": "environment"}

GET /environment/{location_id}?timestamp_utc=...
    → {
        "location_id": "LOC_+27.1230_+88.4560",
        "timestamp_utc": "2026-09-02T12:00:00Z",
        "rainfall_1h": 12.4,
        "rainfall_6h": 48.2,
        "rainfall_24h": 91.5,
        "rainfall_3d": 165.3,
        "rainfall_7d": 240.1,
        "forecast_rainfall": 42.0,
        "moisture_indicator": 0.71,
        "data_source": "open_meteo"
      }

GET /environment/{location_id}/history?hours=72&step_hours=1
    → {"location_id": "...", "history": [...]}
```

Error responses follow `docs/api-contract.md §9`:
```json
{"error": {"code": "INVALID_LOCATION_ID", "message": "..."}}
```

---

## 11. Update Frequency

| Data type | Update frequency |
|---|---|
| ERA5-Land reanalysis (historical) | ~5-day lag; static once ingested |
| ICON NWP forecast | Every 6 hours from Open-Meteo |
| Computed dynamic features | Each pipeline run (configurable via scheduler) |
| Cache TTL | 1 hour (configurable) |

This is **periodic polling**, not continuous monitoring. Do not claim
"real-time" capability — the ERA5-Land source has a ~5-day reanalysis lag.

---

## 12. Cloudburst Project Reuse

No previous cloudburst project was found locally. The following engineering
patterns were implemented following the same design principles described in
the project brief:

| Component | Implementation |
|---|---|
| Provider abstraction | `WeatherProvider` ABC → `OpenMeteoProvider` / `MockWeatherProvider` |
| Weather API client | HTTP GET with exponential backoff (`open_meteo_provider.py`) |
| Rainfall ingestion | `RainfallIngestor` (validation, normalisation, plausibility checks) |
| Time-series aggregation | `rainfall_windows.py` (pure functions, configurable windows) |
| Caching | `LocalCache` (in-memory TTL, thread-safe) |
| Exception handling | `ProviderError` hierarchy (timeout, unavailable, rate limit, etc.) |
| Configuration | `EnvironmentConfig` dataclass (no hard-coded values) |
| UTC normalisation | Wraps `data_engineering.validation.normalize_timestamp` |

---

## 13. Known Limitations

1. **ERA5-Land lag**: Very recent hours (< 5 days) fall back to forecast data,
   which is less accurate for past conditions.
2. **Grid resolution**: ~1 km grid may miss sub-kilometre rainfall variability
   important for local slope hydrology.
3. **Moisture indicator**: Rainfall-derived proxy only. Ignores soil type,
   drainage, evapotranspiration. Use SMAP/Sentinel-1 for production.
4. **No real-time streaming**: Pipeline is batch/polling. Not sub-minute updates.
5. **No spatial interpolation**: Nearest-neighbour mapping only. Bilinear
   interpolation would be more accurate for coarser grids.
6. **No historical climatology**: `rainfall_anomaly` uses a 7-day rolling
   baseline, not a multi-year climatological baseline.
7. **No duplicate timestamp deduplication**: Duplicate timestamps in provider
   data are passed through (logged). Caller should be aware.

---

## 14. File Structure

```
environment/
├── __init__.py                        # Public interface
├── config.py                          # EnvironmentConfig (all configurable values)
├── schemas/
│   ├── __init__.py
│   └── dynamic_record.py              # DynamicRecord, DataProvenance, RawObservation, ForecastRecord
├── providers/
│   ├── __init__.py                    # get_provider() factory
│   ├── base.py                        # WeatherProvider ABC + exception hierarchy
│   ├── mock_provider.py               # Deterministic mock (fixture CSV + generator)
│   └── open_meteo_provider.py         # Free Open-Meteo API (ERA5-Land + ICON)
├── ingestion/
│   ├── __init__.py
│   └── rainfall_ingestor.py           # Validate + normalise raw observations → DataFrame
├── preprocessing/
│   ├── __init__.py
│   ├── timestamp_utils.py             # UTC normalisation (wraps data_engineering)
│   └── spatial_mapper.py             # SpatialMapper + generate_location_id (Engineer 1-compatible)
├── features/
│   ├── __init__.py
│   ├── rainfall_windows.py            # Pure window functions (rolling, intensity, antecedent, anomaly)
│   ├── forecast_features.py           # Forecast alignment + accumulation
│   └── moisture_indicator.py         # Sigmoid moisture proxy (documented limitations)
├── aggregation/
│   ├── __init__.py
│   ├── feature_builder.py             # build_dynamic_features() / get_dynamic_conditions()
│   └── pipeline.py                    # DynamicFeaturePipeline (batch)
├── caching/
│   ├── __init__.py
│   └── local_cache.py                 # In-memory TTL cache (thread-safe)
├── APIs/
│   ├── __init__.py
│   └── router.py                      # FastAPI router for Engineer 6
└── tests/
    ├── __init__.py
    ├── fixtures/
    │   └── synthetic_rainfall.csv     # 8-day hourly data, 2 locations, 384 rows
    ├── test_rainfall_windows.py       # 22 tests — pure window functions
    ├── test_forecast_features.py      # 12 tests — forecast accumulation + alignment
    ├── test_timestamp_utils.py        # 13 tests — UTC normalisation
    ├── test_spatial_mapper.py         # 14 tests — location_id + spatial mapping
    ├── test_provider_mock.py          # 15 tests — mock provider + failure modes
    ├── test_cache.py                  # 15 tests — TTL cache behaviour
    ├── test_feature_builder.py        # 20 tests — build_dynamic_features integration
    ├── test_pipeline.py               # 9 tests — batch pipeline
    └── test_dynamic_output.py         # 14 tests — contract compliance
```
