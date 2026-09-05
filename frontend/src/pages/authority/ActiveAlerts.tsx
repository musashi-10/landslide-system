/**
 * Authority Active Alerts — manage current ACTIVE alerts.
 *
 * Allows authorities to view, acknowledge, and resolve alerts.
 * Uses backend alert IDs to avoid duplicate frontend logic.
 */

import { useState, useEffect, useCallback } from 'react';
import { getAlerts } from '../../services/alertService';
import { updateAlertStatus } from '../../services/alertService';
import { RiskBadge } from '../../components/ui/RiskBadge';
import { LoadingState } from '../../components/ui/LoadingState';
import { ErrorState } from '../../components/ui/ErrorState';
import type { AlertEntry } from '../../types';
import {
  AlertTriangle,
  CheckCircle,
  XCircle,
  Clock,
  RefreshCw,
  Bell,
} from 'lucide-react';

export function ActiveAlerts() {
  const [alerts, setAlerts] = useState<AlertEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [actionLoading, setActionLoading] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      const resp = await getAlerts();
      setAlerts(resp.alerts);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load alerts');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  const handleAction = async (alertId: string, action: 'acknowledge' | 'resolve') => {
    setActionLoading(alertId);
    try {
      const updated = await updateAlertStatus(alertId, action);
      setAlerts((prev) =>
        prev.map((a) => (a.alert_id === alertId ? { ...a, status: updated.status } : a))
      );
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Action failed');
    } finally {
      setActionLoading(null);
    }
  };

  if (loading) return <div className="page-container"><LoadingState message="Loading alerts…" /></div>;
  if (error && alerts.length === 0) return <div className="page-container"><ErrorState message={error} onRetry={load} /></div>;

  const activeAlerts = alerts.filter((a) => a.status === 'ACTIVE');

  return (
    <div className="page-container">
      <div className="dashboard-header">
        <div>
          <h1 className="dashboard-title">Active Alerts</h1>
          <p className="dashboard-subtitle">{activeAlerts.length} alert{activeAlerts.length !== 1 ? 's' : ''} requiring attention</p>
        </div>
        <button className="btn btn--ghost btn--sm" onClick={load}>
          <RefreshCw size={14} /> Refresh
        </button>
      </div>

      {error && (
        <div className="auth-error" role="alert">
          <AlertTriangle size={14} /> {error}
        </div>
      )}

      {activeAlerts.length === 0 ? (
        <div className="empty-state">
          <Bell size={40} />
          <h2>No Active Alerts</h2>
          <p>All alerts have been acknowledged or resolved.</p>
        </div>
      ) : (
        <div className="auth-alert-list">
          {/* Header */}
          <div className="auth-alert-list__header">
            <span>Alert ID</span>
            <span>Location</span>
            <span>Risk Level</span>
            <span>Time (UTC)</span>
            <span>Status</span>
            <span>Actions</span>
          </div>

          {activeAlerts.map((alert) => (
            <div key={alert.alert_id} className={`auth-alert-row auth-alert-row--${alert.risk_level.toLowerCase()}`}>
              <span className="auth-alert-row__id">{alert.alert_id}</span>
              <span className="auth-alert-row__location">{alert.location_id}</span>
              <span>
                <RiskBadge level={alert.risk_level} size="sm" />
              </span>
              <span className="auth-alert-row__time">
                <Clock size={12} />
                {new Date(alert.timestamp_utc).toLocaleString('en-GB', {
                  day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit',
                  timeZone: 'UTC',
                })}
              </span>
              <span>
                <span className="status-badge status-badge--active">Active</span>
              </span>
              <div className="auth-alert-row__actions">
                <button
                  className="btn btn--warning btn--xs"
                  onClick={() => handleAction(alert.alert_id, 'acknowledge')}
                  disabled={actionLoading === alert.alert_id}
                  title="Acknowledge alert"
                >
                  {actionLoading === alert.alert_id ? (
                    <span className="btn-spinner--xs" />
                  ) : (
                    <><CheckCircle size={13} /> Acknowledge</>
                  )}
                </button>
                <button
                  className="btn btn--success btn--xs"
                  onClick={() => handleAction(alert.alert_id, 'resolve')}
                  disabled={actionLoading === alert.alert_id}
                  title="Resolve alert"
                >
                  <XCircle size={13} /> Resolve
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Messages section */}
      {activeAlerts.length > 0 && (
        <div className="alerts-messages">
          <h2 className="section-label">Alert Messages</h2>
          <div className="alert-list">
            {activeAlerts.map((alert) => (
              <div key={alert.alert_id} className={`alert-row alert-row--${alert.risk_level.toLowerCase()}`}>
                <div className="alert-row__left">
                  <RiskBadge level={alert.risk_level} size="sm" />
                </div>
                <div className="alert-row__body">
                  <p className="alert-row__message">{alert.message ?? `${alert.risk_level} risk alert at ${alert.location_id}.`}</p>
                  <div className="alert-row__meta">
                    <Clock size={12} />
                    {new Date(alert.timestamp_utc).toLocaleString('en-GB', { timeZone: 'UTC' })} UTC
                    <span>•</span> {alert.alert_id}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
