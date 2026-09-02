


# Landslide System — Data Contract

## 1. Purpose

All engineers must use the same spatial, temporal, feature, and prediction formats so modules can be integrated without rewriting data pipelines.

## 2. Spatial Standard

Every spatial record must contain:

- location_id
- latitude
- longitude
- geometry when applicable
- CRS metadata for geospatial files

location_id is the primary identifier used to connect data from different modules.

## 3. Time Standard

All timestamps must use:

- Field: timestamp_utc
- Format: ISO 8601
- Timezone: UTC

Example:

2026-09-02T12:00:00Z

## 4. Static Features

Static features describe natural vulnerability.

Standard names:

- elevation
- slope
- aspect
- soil_type
- geology
- land_cover
- drainage_feature
- historical_landslide

## 5. Satellite-Derived Features

Satellite processing may provide:

- vegetation indicators
- land-cover features
- bare/exposed surface indicators
- change indicators
- landslide-scar indicators

Each satellite-derived feature must preserve:

- source
- acquisition_time
- spatial_resolution
- processing_version

## 6. Dynamic Features

Dynamic features describe changing environmental conditions.

Standard names:

- rainfall_1h
- rainfall_6h
- rainfall_24h
- rainfall_3d
- rainfall_7d
- forecast_rainfall
- moisture_indicator

All rainfall values use millimetres (mm).

## 7. Model Input

The ML model must be able to consume:

{
  "location_id": "LOC_001",
  "timestamp_utc": "2026-09-02T12:00:00Z",
  "features": {
    "elevation": 1240.5,
    "slope": 32.4,
    "aspect": 145.0,
    "soil_type": "example",
    "geology": "example",
    "land_cover": "forest",
    "historical_landslide": 1,
    "rainfall_1h": 12.4,
    "rainfall_6h": 48.2,
    "rainfall_24h": 91.5,
    "rainfall_3d": 165.3,
    "rainfall_7d": 240.1,
    "forecast_rainfall": 42.0,
    "moisture_indicator": 0.71
  }
}

## 8. Model Output

Every prediction must follow:

{
  "location_id": "LOC_001",
  "timestamp_utc": "2026-09-02T12:00:00Z",
  "risk_probability": 0.78,
  "risk_level": "HIGH",
  "top_risk_factors": [
    "high_24h_rainfall",
    "steep_slope",
    "high_susceptibility"
  ]
}

## 9. Risk Levels

The system uses:

- LOW
- MODERATE
- HIGH
- CRITICAL

Threshold values must remain configurable.

## 10. Integration Rules

Modules must communicate using:

1. Shared schemas
2. Documented JSON APIs
3. Standardized spatial identifiers
4. Standardized timestamps
5. Versioned datasets

No module should depend directly on another module's private implementation.

## 11. Data Provenance

Every external dataset must document:

- source
- acquisition date/time
- geographic coverage
- spatial resolution
- temporal resolution
- preprocessing performed
- license/usage restrictions

## 12. Missing Data

Missing values must never silently become zero.

Each module must either:

- preserve missing values
- apply a documented imputation method
- explicitly mark the value as unavailable

## 13. Versioning

Any change to this shared schema must be documented before implementation.

Breaking changes require agreement across affected engineers.

## 14. Core Principle

STATIC VULNERABILITY
+
DYNAMIC TRIGGER CONDITIONS
→
AI RISK PREDICTION
→
GIS RISK MAP
→
EXPLAINABILITY
→
EARLY WARNING