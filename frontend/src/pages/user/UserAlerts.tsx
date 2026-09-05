/**
 * User Alerts page — shows active and past alerts for the user's location.
 */

import { useState, useEffect } from 'react';
import { useAuth } from '../../context/AuthContext';
import { getAllAlerts } from '../../services/alertService';
import { RiskBadge } from '../../components/ui/RiskBadge';
import { LoadingState } from '../../components/ui/LoadingState';
import { ErrorState } from '../../components/ui/ErrorState';
import type { AlertEntry } from '../../types';
import { Bell, CheckCircle, Clock, AlertTriangle } from 'lucide-react';

function AlertStatusBadge({ status }: { status: AlertEntry['status'] }) {
  const map = {
    ACTIVE: { label: 'Active', cls: 'status-badge--active' },
    ACKNOWLEDGED: { label: 'Acknowledged', cls: 'status-badge--ack' },
    RESOLVED: { label: 'Resolved', cls: 'status-badge--resolved' },
  };
  const { label, cls } = map[status];
  return <span className={`status-badge ${cls}`}>{label}</span>;
}

export function UserAlerts() {
  const { user } = useAuth();
  const [alerts, setAlerts] = useState<AlertEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = async () => {
    try {
      const resp = await getAllAlerts();
      // Show only alerts for this user's location
      const filtered = user?.location_id
        ? resp.alerts.filter((a) => a.location_id === user.location_id)
        : resp.alerts;
      setAlerts(filtered);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load alerts');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { load(); }, [user?.location_id]);

  if (loading) return <div className="page-container"><LoadingState message="Loading alerts…" /></div>;
  if (error) return <div className="page-container"><ErrorState message={error} onRetry={load} /></div>;

  const active = alerts.filter((a) => a.status === 'ACTIVE');
  const past = alerts.filter((a) => a.status !== 'ACTIVE');

  return (
    <div className="page-container">
      <div className="dashboard-header">
        <h1 className="dashboard-title">My Alerts</h1>
        <p className="dashboard-subtitle">Alert history for location {user?.location_id}</p>
      </div>

      {active.length === 0 && past.length === 0 && (
        <div className="empty-state">
          <Bell size={40} />
          <h2>No Alerts</h2>
          <p>No alerts found for your monitored location.</p>
        </div>
      )}

      {active.length > 0 && (
        <section className="alerts-section">
          <h2 className="section-label">
            <AlertTriangle size={16} className="section-label__icon section-label__icon--warn" />
            Active Warnings ({active.length})
          </h2>
          <div className="alert-list">
            {active.map((alert) => (
              <div key={alert.alert_id} className={`alert-row alert-row--${alert.risk_level.toLowerCase()}`}>
                <div className="alert-row__left">
                  <RiskBadge level={alert.risk_level} size="sm" />
                  <AlertStatusBadge status={alert.status} />
                </div>
                <div className="alert-row__body">
                  <p className="alert-row__message">{alert.message}</p>
                  <div className="alert-row__meta">
                    <Clock size={12} />
                    {new Date(alert.timestamp_utc).toLocaleString('en-GB', { timeZone: 'UTC' })} UTC
                    <span>•</span>
                    {alert.alert_id}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </section>
      )}

      {past.length > 0 && (
        <section className="alerts-section">
          <h2 className="section-label">
            <CheckCircle size={16} className="section-label__icon section-label__icon--ok" />
            Alert History ({past.length})
          </h2>
          <div className="alert-list">
            {past.map((alert) => (
              <div key={alert.alert_id} className="alert-row alert-row--past">
                <div className="alert-row__left">
                  <RiskBadge level={alert.risk_level} size="sm" />
                  <AlertStatusBadge status={alert.status} />
                </div>
                <div className="alert-row__body">
                  <p className="alert-row__message">{alert.message}</p>
                  <div className="alert-row__meta">
                    <Clock size={12} />
                    {new Date(alert.timestamp_utc).toLocaleString('en-GB', { timeZone: 'UTC' })} UTC
                    <span>•</span>
                    {alert.alert_id}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </section>
      )}
    </div>
  );
}
