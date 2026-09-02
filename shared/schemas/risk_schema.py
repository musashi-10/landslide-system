from pydantic import BaseModel, Field


class RiskPrediction(BaseModel):
    location_id: str
    timestamp_utc: str
    risk_probability: float = Field(ge=0.0, le=1.0)
    risk_level: str
    top_risk_factors: list[str] = []
