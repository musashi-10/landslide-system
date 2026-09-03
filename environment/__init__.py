"""
environment — Dynamic Environmental Condition Pipeline

Engineer 4 module.  Branch: engineer-4-environment.

Public interface for Engineer 5 (ML model)
------------------------------------------
>>> from environment import build_dynamic_features, get_dynamic_conditions
>>> from environment.providers import MockWeatherProvider

>>> provider = MockWeatherProvider()
>>> record = build_dynamic_features(
...     location_id="LOC_+27.1230_+088.4560",
...     latitude=27.123,
...     longitude=88.456,
...     timestamp_utc="2026-09-02T12:00:00Z",
...     provider=provider,
... )

Public interface for Engineer 6 (backend / dashboard)
------------------------------------------------------
>>> from environment.APIs import router  # mount in FastAPI app

Pipeline (batch processing)
---------------------------
>>> from environment import DynamicFeaturePipeline, LocationSpec

Feature schemas
---------------
>>> from environment.schemas import DynamicRecord, DataProvenance
"""

from environment.aggregation.feature_builder import (
    build_dynamic_features,
    dynamic_record_to_dict,
    get_dynamic_conditions,
)
from environment.aggregation.pipeline import DynamicFeaturePipeline, LocationSpec, PipelineResult
from environment.config import DEFAULT_CONFIG, EnvironmentConfig
from environment.schemas.dynamic_record import DataProvenance, DynamicRecord

__all__ = [
    # Primary interface
    "build_dynamic_features",
    "get_dynamic_conditions",
    "dynamic_record_to_dict",
    # Batch pipeline
    "DynamicFeaturePipeline",
    "LocationSpec",
    "PipelineResult",
    # Config
    "EnvironmentConfig",
    "DEFAULT_CONFIG",
    # Schemas
    "DynamicRecord",
    "DataProvenance",
]
