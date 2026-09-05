/**
 * Authority Command Center — situational awareness dashboard for disaster management.
 *
 * Shows regional summary counts, active alerts, and priority locations.
 * All numbers come from live backend data.
 */

import { Link } from 'react-router-dom';
import { useRiskMap } from '../../hooks/useRiskMap';
import { useAlerts } from '../../hooks/useAlerts';
import { getRiskSummaryCounts } from '../../services/riskService';
import { StatCard } from '../../components/ui/StatCard';
import { RiskBadge } from '../../components/ui/RiskBadge';
import { LoadingState } from '../../components/ui/LoadingState';
import { ErrorState } from '../../components/ui/ErrorState';
import {
  AlertTriangle,
  Map,
  Activity,
  ShieldAlert,
  Clock,
  TrendingUp,
  ArrowRight,
} from 'lucide-react';
import { RISK_COLORS } from '../../utils/riskColors';

export function CommandCenter() {
  const { data: mapData, loading: mapLoading, error: mapError, reload } = useRiskMap(30_000);
  const { alerts } = useAlerts(15_000);

  const counts = mapData ? getRiskSummaryCounts(mapData) : null;
  const activeAlerts = alerts.length;

  // Priority locations: HIGH and CRITICAL, sorted by probability
  const priorityLocations = mapData
    ? [...mapData.locations]
        .filter((l) => l.risk_level === 'HIGH' || l.risk_level === 'CRITICAL')
        .sort((a, b) => b.risk_probability - a.risk_probability)
        .slice(0, 6)
    : [];

  if (mapLoading && !mapData) {
    return <div className="page-container"><LoadingState message="Loading command center…" size="lg" /></div>;
  }

  if (mapError && !mapData) {
    return <div className="page-container"><ErrorState title="Command Center Unavailable" message={mapError} onRetry={reload} /></div>;
  }

  return (
    <div className="page-container cmd-center">
      <div className="dashboard-header">
        <div>
          <h1 className="dashboard-title">Command Center</h1>
          <p className="dashboard-subtitle">
            Regional landslide risk overview
            {mapData && (
              <> — <Clock size={12} /> {new Date(mapData.timestamp_utc).toLocaleString('en-GB', { timeZone: 'UTC' })} UTC</>
            )}
          </p>
        </div>
        <div className="cmd-center__actions">
          <Link to="/authority/map" className="btn btn--primary btn--sm">
            <Map size={16} /> Open Risk Map
          </Link>
        </div>
      </div>

      {/* ── Summary Stats ─────────────────────────────────────────────── */}
      <section className="cmd-stats">
        <StatCard
          label="Critical Zones"
          value={counts?.CRITICAL ?? '—'}
          icon={<ShieldAlert size={20} />}
          accentColor={RISK_COLORS.CRITICAL.fill}
          onClick={() => window.location.href = '/authority/map?level=CRITICAL'}
        />
        <StatCard
          label="High Risk Zones"
          value={counts?.HIGH ?? '—'}
          icon={<AlertTriangle size={20} />}
          accentColor={RISK_COLORS.HIGH.fill}
          onClick={() => window.location.href = '/authority/map?level=HIGH'}
        />
        <StatCard
          label="Moderate Zones"
          value={counts?.MODERATE ?? '—'}
          icon={<Activity size={20} />}
          accentColor={RISK_COLORS.MODERATE.fill}
        />
        <StatCard
          label="Low Risk Zones"
          value={counts?.LOW ?? '—'}
          icon={<TrendingUp size={20} />}
          accentColor={RISK_COLORS.LOW.fill}
        />
        <StatCard
          label="Active Alerts"
          value={activeAlerts}
          icon={<AlertTriangle size={20} />}
          accentColor="#f97316"
          subtitle="Requiring attention"
          onClick={() => window.location.href = '/authority/alerts'}
        />
      </section>

      {/* ── Priority Locations ───────────────────────────────────────── */}
      <section className="cmd-section">
        <div className="cmd-section__header">
          <h2 className="cmd-section__title">
            <ShieldAlert size={18} />
            Priority Locations
          </h2>
          <Link to="/authority/map" className="cmd-section__link">
            View all on map <ArrowRight size={14} />
          </Link>
        </div>

        {priorityLocations.length === 0 ? (
          <div className="empty-inline">
            No HIGH or CRITICAL zones at this time.
          </div>
        ) : (
          <div className="priority-grid">
            {priorityLocations.map((loc) => {
              const c = RISK_COLORS[loc.risk_level];
              const pct = Math.round(loc.risk_probability * 100);
              return (
                <Link
                  key={loc.location_id}
                  to={`/authority/map`}
                  className="priority-card"
                  style={{ borderLeftColor: c.fill }}
                >
                  <div className="priority-card__header">
                    <span className="priority-card__id">{loc.location_id}</span>
                    <RiskBadge level={loc.risk_level} size="sm" />
                  </div>
                  <div className="priority-card__prob">
                    <span style={{ color: c.fill }}>{pct}%</span>
                    <span className="priority-card__label">probability</span>
                  </div>
                  <div className="priority-card__bar">
                    <div className="priority-card__bar-fill" style={{ width: `${pct}%`, background: c.fill }} />
                  </div>
                </Link>
              );
            })}
          </div>
        )}
      </section>

      {/* ── Active Alerts Preview ─────────────────────────────────────── */}
      <section className="cmd-section">
        <div className="cmd-section__header">
          <h2 className="cmd-section__title">
            <AlertTriangle size={18} />
            Active Alerts
          </h2>
          <Link to="/authority/alerts" className="cmd-section__link">
            Manage all <ArrowRight size={14} />
          </Link>
        </div>

        {alerts.length === 0 ? (
          <div className="empty-inline">No active alerts at this time.</div>
        ) : (
          <div className="alert-list">
            {alerts.slice(0, 4).map((alert) => (
              <div key={alert.alert_id} className={`alert-row alert-row--${alert.risk_level.toLowerCase()}`}>
                <div className="alert-row__left">
                  <RiskBadge level={alert.risk_level} size="sm" />
                </div>
                <div className="alert-row__body">
                  <p className="alert-row__message">{alert.message}</p>
                  <div className="alert-row__meta">
                    <Clock size={12} />
                    {new Date(alert.timestamp_utc).toLocaleString('en-GB', { timeZone: 'UTC' })} UTC
                    <span>•</span> {alert.location_id}
                    <span>•</span> {alert.alert_id}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </section>
    </div>
  );
}
