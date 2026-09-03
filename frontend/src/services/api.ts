/**
 * Typed API client for the Landslide Backend.
 *
 * All fetch() calls are centralised here — never scattered through components.
 * Set VITE_USE_MOCK_API=true to use local mock data instead of the real backend.
 */

import type {
  RiskCurrentResponse,
  RiskMapResponse,
  RiskHistoryResponse,
  RiskFactorsResponse,
  AlertsResponse,
  HealthResponse,
} from '../types/risk';

import {
  MOCK_RISK_MAP,
  MOCK_CURRENT,
  MOCK_HISTORY,
  MOCK_FACTORS,
  MOCK_ALERTS,
} from '../mocks/mockData';

const BASE_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000';
const USE_MOCK = import.meta.env.VITE_USE_MOCK_API === 'true';

async function get<T>(path: string): Promise<T> {
  const resp = await fetch(`${BASE_URL}${path}`);
  if (!resp.ok) {
    const body = await resp.json().catch(() => ({}));
    const msg = body?.error?.message ?? body?.detail ?? `HTTP ${resp.status}`;
    throw new Error(msg);
  }
  return resp.json() as Promise<T>;
}

// ── Public API ────────────────────────────────────────────────────────────────

export async function checkHealth(): Promise<HealthResponse> {
  if (USE_MOCK) return { status: 'ok', service: 'mock' };
  return get<HealthResponse>('/health');
}

export async function getRiskMap(opts?: {
  risk_level?: string;
  bbox?: string;
}): Promise<RiskMapResponse> {
  if (USE_MOCK) return MOCK_RISK_MAP;
  const params = new URLSearchParams();
  if (opts?.risk_level) params.set('risk_level', opts.risk_level);
  if (opts?.bbox) params.set('bbox', opts.bbox);
  const qs = params.toString() ? `?${params}` : '';
  return get<RiskMapResponse>(`/risk/map${qs}`);
}

export async function getCurrentRisk(locationId: string): Promise<RiskCurrentResponse> {
  if (USE_MOCK) {
    return (
      MOCK_CURRENT[locationId] ?? {
        location_id: locationId,
        timestamp_utc: new Date().toISOString(),
        risk_probability: 0.35,
        risk_level: 'MODERATE',
        top_risk_factors: [],
      }
    );
  }
  return get<RiskCurrentResponse>(`/risk/current/${encodeURIComponent(locationId)}`);
}

export async function getRiskHistory(locationId: string): Promise<RiskHistoryResponse> {
  if (USE_MOCK) return { ...MOCK_HISTORY, location_id: locationId };
  return get<RiskHistoryResponse>(`/risk/history/${encodeURIComponent(locationId)}`);
}

export async function getRiskFactors(locationId: string): Promise<RiskFactorsResponse> {
  if (USE_MOCK) return { ...MOCK_FACTORS, location_id: locationId };
  return get<RiskFactorsResponse>(`/risk/factors/${encodeURIComponent(locationId)}`);
}

export async function getAlerts(): Promise<AlertsResponse> {
  if (USE_MOCK) return MOCK_ALERTS;
  return get<AlertsResponse>('/alerts');
}
