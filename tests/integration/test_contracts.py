from shared.schemas import EnvironmentalFeatures, RiskPrediction


def test_environmental_feature_contract():
    data = EnvironmentalFeatures(
        location_id="LOC_001",
        latitude=27.123,
        longitude=88.456,
        timestamp_utc="2026-09-02T12:00:00Z",
        elevation=1240.5,
        slope=32.4,
        aspect=145.0,
        soil_type="example",
        geology="example",
        land_cover="forest",
        historical_landslide=1,
        rainfall_1h=12.4,
        rainfall_6h=48.2,
        rainfall_24h=91.5,
        rainfall_3d=165.3,
        rainfall_7d=240.1,
        forecast_rainfall=42.0,
        moisture_indicator=0.71,
    )

    assert data.location_id == "LOC_001"
    assert data.slope == 32.4
    assert data.rainfall_24h == 91.5


def test_risk_prediction_contract():
    prediction = RiskPrediction(
        location_id="LOC_001",
        timestamp_utc="2026-09-02T12:00:00Z",
        risk_probability=0.78,
        risk_level="HIGH",
        top_risk_factors=[
            "high_24h_rainfall",
            "steep_slope",
            "high_susceptibility",
        ],
    )

    assert prediction.location_id == "LOC_001"
    assert 0.0 <= prediction.risk_probability <= 1.0
    assert prediction.risk_level == "HIGH"
    assert len(prediction.top_risk_factors) > 0
