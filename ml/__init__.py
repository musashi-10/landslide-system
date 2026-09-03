"""
ML Risk Engine
==============

Central landslide risk prediction service.

Quick start (inference)
-----------------------
    from ml.inference import Predictor
    predictor = Predictor.from_registry(model_dir="ml/models/artifacts")
    result = predictor.predict({
        "location_id": "LOC_001",
        "timestamp_utc": "2026-09-02T12:00:00Z",
        "features": { ... }
    })

Quick start (mock, for Engineer 6)
-----------------------------------
    from ml.inference import MockPredictor
    mock = MockPredictor()
    result = mock.predict({"location_id": "LOC_TEST_001", ...})

Run the inference API
---------------------
    uvicorn ml.api.app:app --host 0.0.0.0 --port 8001
"""

__version__ = "1.0.0"
