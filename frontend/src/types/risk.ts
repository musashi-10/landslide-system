/**
 * TypeScript interfaces matching docs/api-contract.md exactly.
 * The frontend must use these types for all API communication.
 */

export type RiskLevel = 'LOW' | 'MODERATE' | 'HIGH' | 'CRITICAL';

export interface HealthResponse {
  status: string;
  service: string;
}

export interface RiskCurrentResponse {
  location_id: string;
  timestamp_utc: string;
  risk_probability: number;
  risk_level: RiskLevel;
  top_risk_factors: string[];
}

export interface RiskMapLocation {
  location_id: string;
  latitude: number;
  longitude: number;
  risk_probability: number;
  risk_level: RiskLevel;
}

export interface RiskMapResponse {
  timestamp_utc: string;
  locations: RiskMapLocation[];
}

export interface RiskHistoryEntry {
  timestamp_utc: string;
  risk_probability: number;
  risk_level: RiskLevel;
}

export interface RiskHistoryResponse {
  location_id: string;
  history: RiskHistoryEntry[];
}

export interface RiskFactor {
  feature: string;
  value: number | null;
  importance: number | null;
}

export interface RiskFactorsResponse {
  location_id: string;
  factors: RiskFactor[];
}

export interface AlertEntry {
  alert_id: string;
  location_id: string;
  timestamp_utc: string;
  risk_level: RiskLevel;
  status: 'ACTIVE' | 'ACKNOWLEDGED' | 'RESOLVED';
  message?: string;
}

export interface AlertsResponse {
  alerts: AlertEntry[];
}

export interface ApiError {
  code: string;
  message: string;
}
