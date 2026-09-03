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
    { feature: 'high_24h_rainfall', value: 91.5, importance: 0.42 },
    { feature: 'steep_slope', value: 32.4, importance: 0.31 },
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
  ],
};
