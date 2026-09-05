/**
 * Mock API data for frontend development without a running backend.
 *
 * Enable by setting VITE_USE_MOCK_API=true in frontend/.env
 * All mock responses use the same schema as production responses.
 */

import type {
  RiskMapResponse,
  RiskCurrentResponse,
  RiskHistoryResponse,
  RiskFactorsResponse,
  AlertsResponse,
} from '../types/risk';

export const MOCK_RISK_MAP: RiskMapResponse = {
  timestamp_utc: '2026-09-03T10:00:00Z',
  locations: [
    { location_id: 'LOC_001', latitude: 27.123, longitude: 88.456, risk_probability: 0.78, risk_level: 'HIGH' },
    { location_id: 'LOC_002', latitude: 27.234, longitude: 88.567, risk_probability: 0.45, risk_level: 'MODERATE' },
    { location_id: 'LOC_003', latitude: 27.345, longitude: 88.678, risk_probability: 0.92, risk_level: 'CRITICAL' },
    { location_id: 'LOC_004', latitude: 27.456, longitude: 88.789, risk_probability: 0.12, risk_level: 'LOW' },
    { location_id: 'LOC_005', latitude: 27.567, longitude: 88.890, risk_probability: 0.67, risk_level: 'HIGH' },
    { location_id: 'LOC_006', latitude: 27.678, longitude: 88.321, risk_probability: 0.30, risk_level: 'MODERATE' },
    { location_id: 'LOC_007', latitude: 27.789, longitude: 88.210, risk_probability: 0.88, risk_level: 'CRITICAL' },
    { location_id: 'LOC_008', latitude: 27.890, longitude: 88.100, risk_probability: 0.08, risk_level: 'LOW' },
    { location_id: 'LOC_009', latitude: 27.200, longitude: 88.350, risk_probability: 0.55, risk_level: 'MODERATE' },
    { location_id: 'LOC_010', latitude: 27.400, longitude: 88.600, risk_probability: 0.73, risk_level: 'HIGH' },
    { location_id: 'LOC_011', latitude: 27.600, longitude: 88.450, risk_probability: 0.19, risk_level: 'LOW' },
    { location_id: 'LOC_012', latitude: 27.500, longitude: 88.300, risk_probability: 0.61, risk_level: 'HIGH' },
  ],
};

export const MOCK_CURRENT: Record<string, RiskCurrentResponse> = {
  LOC_001: {
    location_id: 'LOC_001',
    timestamp_utc: '2026-09-03T10:00:00Z',
    risk_probability: 0.78,
    risk_level: 'HIGH',
    top_risk_factors: ['high_24h_rainfall', 'steep_slope', 'high_susceptibility'],
  },
  LOC_003: {
    location_id: 'LOC_003',
    timestamp_utc: '2026-09-03T10:00:00Z',
    risk_probability: 0.92,
    risk_level: 'CRITICAL',
    top_risk_factors: ['high_24h_rainfall', 'steep_slope', 'historical_landslide_activity'],
  },
  LOC_005: {
    location_id: 'LOC_005',
    timestamp_utc: '2026-09-03T10:00:00Z',
    risk_probability: 0.67,
    risk_level: 'HIGH',
    top_risk_factors: ['high_7d_rainfall', 'high_soil_moisture'],
  },
  LOC_007: {
    location_id: 'LOC_007',
    timestamp_utc: '2026-09-03T10:00:00Z',
    risk_probability: 0.88,
    risk_level: 'CRITICAL',
    top_risk_factors: ['intense_1h_rainfall', 'steep_slope', 'high_susceptibility', 'high_soil_moisture'],
  },
};

export const MOCK_HISTORY: RiskHistoryResponse = {
  location_id: 'LOC_001',
  history: [
    { timestamp_utc: '2026-09-01T12:00:00Z', risk_probability: 0.22, risk_level: 'LOW' },
    { timestamp_utc: '2026-09-01T18:00:00Z', risk_probability: 0.35, risk_level: 'MODERATE' },
    { timestamp_utc: '2026-09-02T06:00:00Z', risk_probability: 0.48, risk_level: 'MODERATE' },
    { timestamp_utc: '2026-09-02T12:00:00Z', risk_probability: 0.55, risk_level: 'MODERATE' },
    { timestamp_utc: '2026-09-02T18:00:00Z', risk_probability: 0.62, risk_level: 'HIGH' },
    { timestamp_utc: '2026-09-03T06:00:00Z', risk_probability: 0.71, risk_level: 'HIGH' },
    { timestamp_utc: '2026-09-03T10:00:00Z', risk_probability: 0.78, risk_level: 'HIGH' },
  ],
};

export const MOCK_FACTORS: RiskFactorsResponse = {
  location_id: 'LOC_001',
  factors: [
    { feature: 'rainfall_24h', value: 91.5, importance: 0.42 },
    { feature: 'slope', value: 32.4, importance: 0.31 },
    { feature: 'rainfall_7d', value: 240.1, importance: 0.12 },
    { feature: 'rainfall_1h', value: 12.4, importance: 0.05 },
    { feature: 'rainfall_6h', value: 48.2, importance: 0.04 },
    { feature: 'rainfall_3d', value: 165.3, importance: 0.03 },
    { feature: 'forecast_rainfall', value: 42.0, importance: 0.02 },
    { feature: 'high_susceptibility', value: 0.78, importance: 0.17 },
    { feature: 'historical_landslide_activity', value: 1, importance: 0.10 },
  ],
};

export const MOCK_ALERTS: AlertsResponse = {
  alerts: [
    {
      alert_id: 'ALT_00000001',
      location_id: 'LOC_003',
      timestamp_utc: '2026-09-03T09:45:00Z',
      risk_level: 'CRITICAL',
      status: 'ACTIVE',
      message: 'CRITICAL landslide risk at LOC_003 (probability 92%). Immediate attention required.',
    },
    {
      alert_id: 'ALT_00000002',
      location_id: 'LOC_001',
      timestamp_utc: '2026-09-03T10:00:00Z',
      risk_level: 'HIGH',
      status: 'ACTIVE',
      message: 'HIGH landslide risk at LOC_001 (probability 78%). Immediate attention required.',
    },
    {
      alert_id: 'ALT_00000003',
      location_id: 'LOC_007',
      timestamp_utc: '2026-09-03T09:30:00Z',
      risk_level: 'CRITICAL',
      status: 'ACTIVE',
      message: 'CRITICAL landslide risk at LOC_007 (probability 88%). Immediate attention required.',
    },
    {
      alert_id: 'ALT_00000004',
      location_id: 'LOC_005',
      timestamp_utc: '2026-09-03T08:15:00Z',
      risk_level: 'HIGH',
      status: 'ACTIVE',
      message: 'HIGH landslide risk at LOC_005 (probability 67%). Monitor closely.',
    },
  ],
};

export const MOCK_ALL_ALERTS: AlertsResponse = {
  alerts: [
    ...MOCK_ALERTS.alerts,
    {
      alert_id: 'ALT_00000005',
      location_id: 'LOC_010',
      timestamp_utc: '2026-09-02T14:00:00Z',
      risk_level: 'HIGH',
      status: 'ACKNOWLEDGED',
      message: 'HIGH landslide risk at LOC_010 acknowledged by authority.',
    },
    {
      alert_id: 'ALT_00000006',
      location_id: 'LOC_002',
      timestamp_utc: '2026-09-01T22:00:00Z',
      risk_level: 'HIGH',
      status: 'RESOLVED',
      message: 'HIGH risk at LOC_002 resolved. Risk has decreased to MODERATE.',
    },
    {
      alert_id: 'ALT_00000007',
      location_id: 'LOC_009',
      timestamp_utc: '2026-09-01T16:00:00Z',
      risk_level: 'CRITICAL',
      status: 'RESOLVED',
      message: 'CRITICAL risk at LOC_009 resolved after rainfall subsided.',
    },
  ],
};
