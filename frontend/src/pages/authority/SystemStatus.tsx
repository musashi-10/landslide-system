/**
 * Authority System Status — health monitoring for backend services.
 *
 * Shows actual health status from backend endpoints.
 * Uses UNKNOWN rather than falsely displaying ONLINE when data is unavailable.
 */

import { useState, useEffect } from 'react';
import { getSystemHealth } from '../../services/userService';
import { LoadingState } from '../../components/ui/LoadingState';
import type { ServiceHealth, SystemHealthResponse } from '../../types';
import {
  CheckCircle,
  XCircle,
  HelpCircle,
  RefreshCw,
  Wifi,
  Clock,
  Activity,
} from 'lucide-react';

function StatusIcon({ status }: { status: ServiceHealth['status'] }) {
  if (status === 'ONLINE') return <CheckCircle size={18} className="health-icon health-icon--online" />;
  if (status === 'OFFLINE') return <XCircle size={18} className="health-icon health-icon--offline" />;
  return <HelpCircle size={18} className="health-icon health-icon--unknown" />;
}

export function SystemStatus() {
  const [health, setHealth] = useState<SystemHealthResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [lastChecked, setLastChecked] = useState<Date | null>(null);

  const load = async () => {
    setLoading(true);
    try {
      const data = await getSystemHealth();
      setHealth(data);
      setLastChecked(new Date());
    } catch {
      // Keep showing previous data; mark overall as unknown
      setHealth((prev) => prev ?? {
        overall_status: 'UNKNOWN',
        services: [],
      });
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { load(); }, []);

  // Refresh every 60s
  useEffect(() => {
    const id = setInterval(load, 60_000);
    return () => clearInterval(id);
  }, []);

  if (loading && !health) return <div className="page-container"><LoadingState message="Checking system health…" /></div>;

  const overallCls = health?.overall_status === 'ONLINE'
    ? 'overall-status--online'
    : health?.overall_status === 'OFFLINE'
    ? 'overall-status--offline'
    : 'overall-status--unknown';

  return (
    <div className="page-container">
      <div className="dashboard-header">
        <div>
          <h1 className="dashboard-title">System Status</h1>
          {lastChecked && (
            <p className="dashboard-subtitle">
              <Clock size={12} /> Last checked: {lastChecked.toLocaleTimeString('en-GB', { timeZone: 'UTC' })} UTC
            </p>
          )}
        </div>
        <button className="btn btn--ghost btn--sm" onClick={load} disabled={loading}>
          <RefreshCw size={14} className={loading ? 'spin' : ''} /> Refresh
        </button>
      </div>

      {/* Overall Status */}
      <div className={`overall-status ${overallCls}`}>
        <Wifi size={24} />
        <div>
          <span className="overall-status__label">Overall System</span>
          <span className="overall-status__value">{health?.overall_status ?? 'UNKNOWN'}</span>
        </div>
      </div>

      {/* Service Grid */}
      <div className="health-grid">
        {health?.services.map((svc) => (
          <div key={svc.name} className={`health-card health-card--${svc.status.toLowerCase()}`}>
            <div className="health-card__header">
              <StatusIcon status={svc.status} />
              <span className="health-card__name">{svc.name}</span>
            </div>
            <span className={`health-card__status health-card__status--${svc.status.toLowerCase()}`}>
              {svc.status}
            </span>
            {svc.latency_ms !== undefined && (
              <div className="health-card__latency">
                <Activity size={12} />
                {svc.latency_ms}ms
              </div>
            )}
            {svc.last_check_utc && (
              <div className="health-card__time">
                <Clock size={11} />
                {new Date(svc.last_check_utc).toLocaleTimeString('en-GB', { timeZone: 'UTC' })} UTC
              </div>
            )}
          </div>
        ))}

        {(!health?.services || health.services.length === 0) && (
          <div className="empty-inline">
            System health data unavailable. Status is UNKNOWN.
          </div>
        )}
      </div>

      <p className="health-disclaimer">
        System health is indicative only. UNKNOWN status means health data could not be retrieved — it does not confirm ONLINE state.
      </p>
    </div>
  );
}
