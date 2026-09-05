/**
 * User Dashboard — citizen's primary monitoring view.
 *
 * Shows current risk, probability, explanation, rainfall, and history
 * for the user's registered location.
 */



import { useAuth } from '../../context/AuthContext';
import { useCurrentRisk } from '../../hooks/useCurrentRisk';
import { useAlerts } from '../../hooks/useAlerts';
import { RiskBadge } from '../../components/ui/RiskBadge';
import { RiskFactorList } from '../../components/ui/RiskFactorList';
import { RainfallCard } from '../../components/ui/RainfallCard';
import { SkeletonCard } from '../../components/ui/LoadingState';
import { ErrorState } from '../../components/ui/ErrorState';
import { HistoryChart } from '../../components/HistoryChart/HistoryChart';
import { extractRainfall, computeRiskDelta } from '../../services/riskService';
import { RISK_COLORS } from '../../utils/riskColors';
import {
  AlertTriangle,
  TrendingUp,
  TrendingDown,
  Minus,
  MapPin,
  Clock,
} from 'lucide-react';

export function UserDashboard() {
  const { user } = useAuth();
  const locationId = user?.location_id ?? null;

  const { current, history, factors, loading, error } = useCurrentRisk(locationId);
  const { alerts } = useAlerts(15_000);

  // User-specific active alert
  const userAlert = alerts.find((a) => a.location_id === locationId);

  const pct = current ? Math.round(current.risk_probability * 100) : 0;
  const riskColor = current ? RISK_COLORS[current.risk_level] : null;

  const rainfall = factors ? extractRainfall(factors) : null;
  const delta = history ? computeRiskDelta(history.history) : null;

  if (!locationId) {
    return (
      <div className="page-container">
        <div className="empty-state">
          <MapPin size={40} />
          <h2>No Location Set</h2>
          <p>Please update your settings to select a monitored location.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="page-container user-dashboard">
      {/* ── Critical Alert Banner ─────────────────────────────────── */}
      {userAlert && (userAlert.risk_level === 'CRITICAL' || userAlert.risk_level === 'HIGH') && (
        <div className={`crit-banner crit-banner--${userAlert.risk_level.toLowerCase()}`} role="alert" aria-live="assertive">
          <AlertTriangle size={20} className="crit-banner__icon" />
          <div className="crit-banner__body">
            <strong>
              {userAlert.risk_level === 'CRITICAL' ? '🚨 CRITICAL LANDSLIDE WARNING' : '⚠️ HIGH RISK WARNING'}
            </strong>
            <p>
              Your monitored location is currently classified as{' '}
              <strong>{userAlert.risk_level}</strong> risk.
              {current && <> Risk Probability: <strong>{pct}%</strong>.</>}
            </p>
            <p>Avoid unnecessary travel through exposed landslide-prone areas. Follow official guidance.</p>
            <small>
              <Clock size={12} />
              {' '}Last updated: {current ? new Date(current.timestamp_utc).toLocaleString('en-GB', { timeZone: 'UTC' }) + ' UTC' : '—'}
            </small>
          </div>
        </div>
      )}

      <div className="dashboard-header">
        <div>
          <h1 className="dashboard-title">My Location Monitor</h1>
          <div className="dashboard-location">
            <MapPin size={14} />
            <span>{locationId}</span>
          </div>
        </div>
      </div>

      {loading && !current ? (
        <div className="dashboard-grid">
          <SkeletonCard />
          <SkeletonCard />
          <SkeletonCard />
        </div>
      ) : error ? (
        <ErrorState
          title="Risk data temporarily unavailable"
          message={error}
          lastUpdated={null}
        />
      ) : current ? (
        <div className="dashboard-grid">
          {/* ── Risk Card ────────────────────────────────────────── */}
          <div className="risk-current-card" style={{ borderTopColor: riskColor?.fill }}>
            <div className="risk-current-card__header">
              <span className="risk-current-card__label">CURRENT LANDSLIDE RISK</span>
              <RiskBadge level={current.risk_level} size="lg" />
            </div>

            <div className="risk-current-card__prob">
              <span className="risk-prob-number" style={{ color: riskColor?.fill }}>
                {pct}%
              </span>
              <span className="risk-prob-label">probability</span>
            </div>

            <div
              className="prob-bar"
              role="progressbar"
              aria-valuenow={pct}
              aria-valuemin={0}
              aria-valuemax={100}
            >
              <div
                className="prob-fill"
                style={{ width: `${pct}%`, background: riskColor?.fill }}
              />
            </div>

            <p className="risk-current-card__disclaimer">
              Model-based risk estimate — not a guarantee of landslide occurrence.
            </p>

            <div className="risk-current-card__updated">
              <Clock size={12} />
              Updated: {new Date(current.timestamp_utc).toLocaleString('en-GB', { timeZone: 'UTC' })} UTC
            </div>
          </div>

          {/* ── What Changed ──────────────────────────────────────── */}
          {delta && (
            <div className="change-card">
              <h3 className="change-card__title">WHAT CHANGED?</h3>
              <div className="change-items">
                <div className="change-item">
                  {delta.delta > 0 ? (
                    <TrendingUp size={16} className="change-item__icon change-item__icon--up" />
                  ) : delta.delta < 0 ? (
                    <TrendingDown size={16} className="change-item__icon change-item__icon--down" />
                  ) : (
                    <Minus size={16} className="change-item__icon" />
                  )}
                  <span className="change-item__label">Risk Probability</span>
                  <span className={`change-item__value ${delta.delta > 0 ? 'change-item__value--up' : delta.delta < 0 ? 'change-item__value--down' : ''}`}>
                    {delta.delta > 0 ? '+' : ''}{delta.delta}%
                  </span>
                </div>
                <div className="change-item">
                  <MapPin size={16} className="change-item__icon" />
                  <span className="change-item__label">Risk Level</span>
                  <span className="change-item__value">
                    {delta.previousLevel} → {delta.currentLevel}
                  </span>
                </div>
              </div>
            </div>
          )}

          {/* ── Risk Factors ──────────────────────────────────────── */}
          {factors && factors.factors.length > 0 && (
            <div className="dashboard-card">
              <RiskFactorList
                factors={factors.factors}
                riskLevel={current.risk_level}
                title={`Why is my risk ${current.risk_level.toLowerCase()}?`}
              />
            </div>
          )}

          {/* ── Rainfall ─────────────────────────────────────────── */}
          {rainfall && (
            <div className="dashboard-card">
              <RainfallCard data={rainfall} />
            </div>
          )}

          {/* ── History Chart ─────────────────────────────────────── */}
          {history && history.history.length > 1 && (
            <div className="dashboard-card dashboard-card--wide">
              <HistoryChart history={history.history} locationId={locationId} />
            </div>
          )}
        </div>
      ) : null}
    </div>
  );
}
