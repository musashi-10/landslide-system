/**
 * Risk service — wraps existing api.ts functions with additional utility.
 *
 * This module re-exports the core API calls and adds derived helpers
 * (e.g., extracting rainfall data from factors, computing risk deltas).
 */

import {
  getRiskMap,
  getCurrentRisk,
  getRiskHistory,
  getRiskFactors,
  checkHealth,
} from './api';

import type {
  RiskMapResponse,
  RiskHistoryEntry,
  RainfallData,
  RiskFactorsResponse,
} from '../types';

// ── Re-export core API calls ────────────────────────────────────────────────

export { getRiskMap, getCurrentRisk, getRiskHistory, getRiskFactors, checkHealth };

// ── Derived helpers ─────────────────────────────────────────────────────────

const RAINFALL_KEYS = [
  'rainfall_1h',
  'rainfall_6h',
  'rainfall_24h',
  'rainfall_3d',
  'rainfall_7d',
  'forecast_rainfall',
] as const;

/**
 * Extract rainfall data from a RiskFactorsResponse.
 * The backend stores rainfall as regular risk factors — we pull them out
 * into a structured object for the RainfallCard component.
 */
export function extractRainfall(factors: RiskFactorsResponse): RainfallData {
  const result: RainfallData = {
    rainfall_1h: null,
    rainfall_6h: null,
    rainfall_24h: null,
    rainfall_3d: null,
    rainfall_7d: null,
    forecast_rainfall: null,
  };

  for (const f of factors.factors) {
    const key = f.feature.replace('high_', '').replace('intense_', '') as keyof RainfallData;
    if (RAINFALL_KEYS.includes(key as typeof RAINFALL_KEYS[number])) {
      result[key as keyof RainfallData] = f.value;
    }
  }
  return result;
}

/**
 * Compute risk change between two history entries.
 */
export function computeRiskDelta(
  history: RiskHistoryEntry[]
): { delta: number; previousLevel: string; currentLevel: string } | null {
  if (history.length < 2) return null;
  const prev = history[history.length - 2];
  const curr = history[history.length - 1];
  return {
    delta: Math.round((curr.risk_probability - prev.risk_probability) * 100),
    previousLevel: prev.risk_level,
    currentLevel: curr.risk_level,
  };
}

/**
 * Get risk summary counts from map data (for Command Center stats).
 */
export function getRiskSummaryCounts(mapData: RiskMapResponse) {
  const counts = { LOW: 0, MODERATE: 0, HIGH: 0, CRITICAL: 0 };
  for (const loc of mapData.locations) {
    if (loc.risk_level in counts) {
      counts[loc.risk_level as keyof typeof counts]++;
    }
  }
  return counts;
}
