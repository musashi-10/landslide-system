/**
 * User service — notification preferences and profile management.
 *
 * Fully mocked until backend endpoints exist.
 */

import type { NotificationPreferences, SystemHealthResponse } from '../types';

const BASE_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000';
const USE_MOCK = import.meta.env.VITE_USE_MOCK_API === 'true';

function delay(ms = 300): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

// ── Notification preferences ────────────────────────────────────────────────

const DEFAULT_PREFS: NotificationPreferences = {
  critical_sms: true,
  high_sms: false,
  risk_increase: true,
  emergency_alerts: true,
  mobile_number: '+91-9876543210',
};

let mockPrefs: NotificationPreferences = { ...DEFAULT_PREFS };

export async function getNotificationPreferences(): Promise<NotificationPreferences> {
  if (USE_MOCK) {
    await delay();
    return { ...mockPrefs };
  }
  const res = await fetch(`${BASE_URL}/user/notifications`);
  if (!res.ok) throw new Error('Failed to load notification preferences');
  return res.json();
}

export async function saveNotificationPreferences(
  prefs: NotificationPreferences
): Promise<NotificationPreferences> {
  if (USE_MOCK) {
    await delay(500);
    mockPrefs = { ...prefs };
    return { ...mockPrefs };
  }
  const res = await fetch(`${BASE_URL}/user/notifications`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(prefs),
  });
  if (!res.ok) throw new Error('Failed to save notification preferences');
  return res.json();
}

// ── System health ───────────────────────────────────────────────────────────

export async function getSystemHealth(): Promise<SystemHealthResponse> {
  if (USE_MOCK) {
    await delay();
    return {
      overall_status: 'ONLINE',
      services: [
        { name: 'ML Risk Engine', status: 'ONLINE', latency_ms: 45, last_check_utc: new Date().toISOString() },
        { name: 'Rainfall Pipeline', status: 'ONLINE', latency_ms: 120, last_check_utc: new Date().toISOString() },
        { name: 'GIS Service', status: 'ONLINE', latency_ms: 80, last_check_utc: new Date().toISOString() },
        { name: 'Satellite Features', status: 'UNKNOWN', last_check_utc: new Date().toISOString() },
        { name: 'Alert Engine', status: 'ONLINE', latency_ms: 30, last_check_utc: new Date().toISOString() },
        { name: 'Database', status: 'ONLINE', latency_ms: 12, last_check_utc: new Date().toISOString() },
      ],
    };
  }
  const res = await fetch(`${BASE_URL}/health/detailed`);
  if (!res.ok) throw new Error('Failed to load system health');
  return res.json();
}
