# Landslide System — API Contract

## 1. Purpose

This document defines the interfaces between the ML engine, backend, GIS dashboard, and alert system.

All services must follow these contracts.

## 2. Health Check

GET /health

Response:

{
  "status": "ok",
  "service": "service-name"
}

## 3. Current Risk

GET /risk/current/{location_id}

Response:

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

## 4. Risk Map

GET /risk/map

Optional query parameters:

- time
- risk_level
- bounding_box

Response:

{
  "timestamp_utc": "2026-09-02T12:00:00Z",
  "locations": [
    {
      "location_id": "LOC_001",
      "latitude": 27.123,
      "longitude": 88.456,
      "risk_probability": 0.78,
      "risk_level": "HIGH"
    }
  ]
}

## 5. Risk History

GET /risk/history/{location_id}

Response:

{
  "location_id": "LOC_001",
  "history": [
    {
      "timestamp_utc": "2026-09-01T12:00:00Z",
      "risk_probability": 0.42,
      "risk_level": "MODERATE"
    },
    {
      "timestamp_utc": "2026-09-02T12:00:00Z",
      "risk_probability": 0.78,
      "risk_level": "HIGH"
    }
  ]
}

## 6. Risk Factors

GET /risk/factors/{location_id}

Response:

{
  "location_id": "LOC_001",
  "factors": [
    {
      "feature": "rainfall_24h",
      "value": 91.5,
      "importance": 0.42
    },
    {
      "feature": "slope",
      "value": 32.4,
      "importance": 0.31
    }
  ]
}

## 7. Alerts

GET /alerts

Response:

{
  "alerts": [
    {
      "alert_id": "ALT_001",
      "location_id": "LOC_001",
      "timestamp_utc": "2026-09-02T12:00:00Z",
      "risk_level": "HIGH",
      "status": "ACTIVE"
    }
  ]
}

## 8. Prediction Interface

POST /predict

Request:

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

Response:

{
  "location_id": "LOC_001",
  "timestamp_utc": "2026-09-02T12:00:00Z",
  "risk_probability": 0.78,
  "risk_level": "HIGH",
  "top_risk_factors": [
    "high_24h_rainfall",
    "steep_slope",
    "high-susceptibility"
  ]
}

## 9. Error Format

All APIs should return:

{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable explanation"
  }
}

## 10. Versioning

Current API version:

v1

Future breaking changes must use a new API version.

## 11. Integration Rules

The frontend must communicate with the backend through documented APIs.

The backend must communicate with ML inference through the prediction contract.

The alert system must consume standardized risk outputs.

Frontend code must not directly access the ML model internals.

## 12. Mock Data

During parallel development, every API must support mock/sample responses following this contract.

Mock responses must use the same schema as production responses.