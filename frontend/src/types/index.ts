/**
 * Extended TypeScript interfaces for the full platform.
 * 
 * Risk/alert types are re-exported from risk.ts (which mirrors the backend contract).
 * Auth, User, and Notification types are added for frontend-managed features.
 */

export type { 
  RiskLevel, 
  HealthResponse, 
  RiskCurrentResponse, 
  RiskMapLocation, 
  RiskMapResponse, 
  RiskHistoryEntry, 
  RiskHistoryResponse, 
  RiskFactor, 
  RiskFactorsResponse, 
  AlertEntry, 
  AlertsResponse, 
  ApiError 
} from './risk';

// ── Auth ────────────────────────────────────────────────────────────────────

export type UserRole = 'user' | 'authority';

export interface LoginRequest {
  email: string;
  password: string;
}

export interface RegisterRequest {
  name: string;
  email: string;
  mobile: string;
  password: string;
  location_id?: string;
}

export interface AuthResponse {
  token: string;
  user: UserProfile;
}

export interface UserProfile {
  id: string;
  name: string;
  email: string;
  mobile: string;
  role: UserRole;
  location_id: string | null;
}

// ── Notification Preferences ────────────────────────────────────────────────

export interface NotificationPreferences {
  critical_sms: boolean;
  high_sms: boolean;
  risk_increase: boolean;
  emergency_alerts: boolean;
  mobile_number: string;
}

// ── System Health ───────────────────────────────────────────────────────────

export type ServiceStatus = 'ONLINE' | 'OFFLINE' | 'UNKNOWN';

export interface ServiceHealth {
  name: string;
  status: ServiceStatus;
  latency_ms?: number;
  last_check_utc?: string;
}

export interface SystemHealthResponse {
  services: ServiceHealth[];
  overall_status: ServiceStatus;
}

// ── Rainfall (derived from backend features) ────────────────────────────────

export interface RainfallData {
  rainfall_1h: number | null;
  rainfall_6h: number | null;
  rainfall_24h: number | null;
  rainfall_3d: number | null;
  rainfall_7d: number | null;
  forecast_rainfall: number | null;
}

// ── Alert Management (authority) ────────────────────────────────────────────

export type AlertAction = 'acknowledge' | 'resolve';

export interface AlertActionRequest {
  alert_id: string;
  action: AlertAction;
}
