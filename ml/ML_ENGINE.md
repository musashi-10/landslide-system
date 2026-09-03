# ML Risk Engine — Handoff Guide

## Overview

The `ml/` package provides the central landslide risk prediction engine. It bridges the standardised feature outputs from the Data Engineering, GIS, and Satellite pipelines to generate real-time risk probabilities and human-readable feature importance.

As **Engineer 6 (Frontend & Alerting)**, you are the primary consumer of this package.

## Components Provided

### 1. `Predictor` (Production Inference)
Located in `ml/inference/predictor.py`. 
It loads the trained model (sklearn/XGBoost) and the preprocessor from the `ml/models/artifacts/` registry *once* at startup. It takes a raw features dictionary (matching `api-contract.md`) and handles all feature engineering, imputation, scaling, classification, and explainability.

**Usage:**
```python
from ml.inference.predictor import Predictor

# Loads the latest artifact from ml/models/artifacts
predictor = Predictor.from_registry() 

result = predictor.predict({
    "location_id": "LOC_001",
    "timestamp_utc": "2026-09-02T12:00:00Z",
    "features": {
        "elevation": 1240.5,
        "rainfall_24h": 91.5,
        ...
    }
})
# Returns a RiskPrediction pydantic schema matching the API contract
```

### 2. `MockPredictor` (Dashboard Development)
Located in `ml/inference/mock_predictor.py`.
Since you likely don't have a fully trained production model locally yet, you can use the `MockPredictor`. It does not require any model files but returns structurally identical `RiskPrediction` objects. It is deterministic (the same `location_id` always returns the same risk level).

**Usage:**
```python
from ml.inference.mock_predictor import MockPredictor

mock = MockPredictor()
result = mock.predict({"location_id": "LOC_TEST_001", "timestamp_utc": "...", "features": {}})
```

**Known Mock Locations:**
- `LOC_TEST_001` → HIGH risk
- `LOC_TEST_002` → MODERATE risk
- `LOC_TEST_003` → CRITICAL risk
- `LOC_TEST_004` → LOW risk
- *(Any other ID)* → MODERATE risk

### 3. FastAPI Service
Located in `ml/api/app.py`.
This is a fully compliant implementation of the REST API specified in `docs/api-contract.md`. 
If a trained model is missing, **the service automatically falls back to `MockPredictor`**. This means you can run the API right now and start building the frontend against it.

**Run the service:**
```bash
uvicorn ml.api.app:app --host 0.0.0.0 --port 8001
```

Endpoints provided:
- `GET /health` (returns `{"mock_mode": true}` if no model is loaded)
- `POST /predict`
- `GET /risk/current/{location_id}`
- `GET /risk/map`
- `GET /risk/history/{location_id}`
- `GET /risk/factors/{location_id}`

### 4. Risk Level Thresholds (PROVISIONAL)
The mapping from probability [0.0 - 1.0] to a risk level (`LOW`, `MODERATE`, `HIGH`, `CRITICAL`) is handled by `ml.inference.risk_classifier.RiskClassifier`. 
The default thresholds are **provisional** and must be scientifically validated before deployment. They are configurable via `RiskThresholds`.

### 5. Explainability (SHAP)
The API response includes `top_risk_factors`. This is powered by `ml.explainability.explainer.LandslideExplainer` using SHAP values (if `shap` is installed) to ensure theoretically sound feature importance, or falling back to tree-based importance. The internal feature names are automatically mapped to human-readable labels (e.g., `rainfall_24h` → `high_24h_rainfall`).

## Error Handling
All inference errors are caught and raised as a `PredictionError` with a standard `{"error": {"code": "...", "message": "..."}}` schema, preventing dashboard crashes.

## Testing Status
The `ml/` package is fully unit-tested (140 tests). You can run them via:
```bash
python -m pytest ml/tests/
```
