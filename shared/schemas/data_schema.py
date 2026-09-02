from pydantic import BaseModel, Field
from typing import Optional


class SpatialRecord(BaseModel):
    location_id: str
    latitude: float
    longitude: float
    geometry: Optional[str] = None


class EnvironmentalFeatures(SpatialRecord):
    timestamp_utc: Optional[str] = None

    elevation: Optional[float] = None
    slope: Optional[float] = None
    aspect: Optional[float] = None

    soil_type: Optional[str] = None
    geology: Optional[str] = None
    land_cover: Optional[str] = None

    historical_landslide: Optional[int] = None

    rainfall_1h: Optional[float] = None
    rainfall_6h: Optional[float] = None
    rainfall_24h: Optional[float] = None
    rainfall_3d: Optional[float] = None
    rainfall_7d: Optional[float] = None

    forecast_rainfall: Optional[float] = None
    moisture_indicator: Optional[float] = None
