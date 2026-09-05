/**
 * Authority Alert History — all resolved and acknowledged alerts.
 */

import { useState, useEffect } from 'react';
import { getAllAlerts } from '../../services/alertService';
import { RiskBadge } from '../../components/ui/RiskBadge';
import { LoadingState } from '../../components/ui/LoadingState';
import { ErrorState } from '../../components/ui/ErrorState';
import type { AlertEntry } from '../../types';
import { Clock, Filter } from 'lucide-react';

function StatusBadge({ status }: { status: AlertEntry['status'] }) {
  const map = {
    ACTIVE: { label: 'Active', cls: 'status-badge--active' },
    ACKNOWLEDGED: { label: 'Acknowledged', cls: 'status-badge--ack' },
    RESOLVED: { label: 'Resolved', cls: 'status-badge--resolved' },
  };
  const { label, cls } = map[status];
  return <span className={`status-badge ${cls}`}>{label}</span>;
}

export function AlertHistory() {
  const [alerts, setAlerts] = useState<AlertEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [statusFilter, setStatusFilter] = useState<string>('all');

  const load = async () => {
    try {
      const resp = await getAllAlerts();
      // History = non-active
      setAlerts(resp.alerts.filter((a) => a.status !== 'ACTIVE'));
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load alert history');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { load(); }, []);

  if (loading) return <div className="page-container"><LoadingState message="Loading history…" /></div>;
  if (error) return <div className="page-container"><ErrorState message={error} onRetry={load} /></div>;

  const filtered = statusFilter === 'all'
    ? alerts
    : alerts.filter((a) => a.status === statusFilter);

  return (
    <div className="page-container">
      <div className="dashboard-header">
        <div>
          <h1 className="dashboard-title">Alert History</h1>
          <p className="dashboard-subtitle">{filtered.length} historical alert{filtered.length !== 1 ? 's' : ''}</p>
        </div>
        <div className="filter-select-wrap">
          <Filter size={14} />
          <select className="filter-select" value={statusFilter} onChange={(e) => setStatusFilter(e.target.value)}>
            <option value="all">All Statuses</option>
            <option value="ACKNOWLEDGED">Acknowledged</option>
            <option value="RESOLVED">Resolved</option>
          </select>
        </div>
      </div>

      {filtered.length === 0 ? (
        <div className="empty-inline">No historical alerts matching the current filter.</div>
      ) : (
        <div className="auth-alert-list">
          <div className="auth-alert-list__header">
            <span>Alert ID</span>
            <span>Location</span>
            <span>Risk Level</span>
            <span>Time (UTC)</span>
            <span>Status</span>
          </div>
          {filtered.map((alert) => (
            <div key={alert.alert_id} className="auth-alert-row auth-alert-row--past">
              <span className="auth-alert-row__id">{alert.alert_id}</span>
              <span className="auth-alert-row__location">{alert.location_id}</span>
              <span><RiskBadge level={alert.risk_level} size="sm" /></span>
              <span className="auth-alert-row__time">
                <Clock size={12} />
                {new Date(alert.timestamp_utc).toLocaleString('en-GB', {
                  day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit',
                  timeZone: 'UTC',
                })}
              </span>
              <span><StatusBadge status={alert.status} /></span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
