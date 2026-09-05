/**
 * Alert service — wraps existing alert API with management actions.
 *
 * Alert acknowledgement/resolution will be mocked until backend endpoints exist.
 */

import { getAlerts } from './api';
import type { AlertEntry, AlertsResponse, AlertAction } from '../types';

const BASE_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000';
const USE_MOCK = import.meta.env.VITE_USE_MOCK_API === 'true';

// ── Re-export ───────────────────────────────────────────────────────────────

export { getAlerts };

// ── Fetch all alerts (including non-active for history) ─────────────────────

export async function getAllAlerts(): Promise<AlertsResponse> {
  if (USE_MOCK) {
    const { MOCK_ALL_ALERTS } = await import('../mocks/mockData');
    return MOCK_ALL_ALERTS;
  }
  const res = await fetch(`${BASE_URL}/alerts?all=true`);
  if (!res.ok) throw new Error('Failed to fetch alert history');
  return res.json();
}

// ── Alert actions (authority only) ──────────────────────────────────────────

export async function updateAlertStatus(
  alertId: string,
  action: AlertAction
): Promise<AlertEntry> {
  if (USE_MOCK) {
    await new Promise((r) => setTimeout(r, 300));
    const newStatus = action === 'acknowledge' ? 'ACKNOWLEDGED' : 'RESOLVED';
    return {
      alert_id: alertId,
      location_id: 'LOC_001',
      timestamp_utc: new Date().toISOString(),
      risk_level: 'HIGH',
      status: newStatus as AlertEntry['status'],
      message: `Alert ${alertId} has been ${newStatus.toLowerCase()}.`,
    };
  }

  const res = await fetch(`${BASE_URL}/alerts/${alertId}/${action}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
  });
  if (!res.ok) throw new Error(`Failed to ${action} alert`);
  return res.json();
}
