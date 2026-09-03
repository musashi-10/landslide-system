import type {
  RiskCurrentResponse,
  RiskHistoryResponse,
  RiskFactorsResponse,
} from '../../types/risk';
import { riskBadgeStyle, RISK_COLORS } from '../../utils/riskColors';
import { HistoryChart } from '../HistoryChart/HistoryChart';

const FACTOR_LABELS: Record<string, string> = {
  high_24h_rainfall: '🌧️ Heavy recent rainfall',
  high_7d_rainfall: '🌧️ High 7-day rainfall',
  steep_slope: '⛰️ Steep slope',
  high_susceptibility: '📍 High susceptibility',
  historical_landslide_activity: '📋 Historical landslide activity',
  high_soil_moisture: '💧 High soil moisture',
  intense_1h_rainfall: '⛈️ Intense recent rainfall',
};

interface Props {
  locationId: string;
  current: RiskCurrentResponse | null;
  history: RiskHistoryResponse | null;
  factors: RiskFactorsResponse | null;
  loading: boolean;
  error: string | null;
}

export function RiskPanel({ locationId, current, history, factors, loading, error }: Props) {
  if (loading) {
    return (
      <div className="risk-panel risk-panel--loading">
        <div className="spinner" aria-label="Loading risk data" />
        <p>Loading risk data…</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="risk-panel risk-panel--error">
        <span className="error-icon">⚠️</span>
        <h3>Risk data unavailable</h3>
        <p className="error-msg">{error}</p>
      </div>
    );
  }

  if (!current) {
    return (
      <div className="risk-panel risk-panel--empty">
        <span className="empty-icon">📍</span>
        <p>Select a location on the map to view risk details</p>
      </div>
    );
  }

  const c = RISK_COLORS[current.risk_level];
  const pct = Math.round(current.risk_probability * 100);

  return (
    <div className="risk-panel" id={`risk-panel-${locationId}`}>
      {/* Header */}
      <div className="risk-panel__header">
        <h3 className="risk-panel__location">{locationId}</h3>
        <span className="risk-badge" style={riskBadgeStyle(current.risk_level)}>
          {current.risk_level}
        </span>
      </div>

      {/* Probability gauge */}
      <div className="risk-panel__prob">
        <div
          className="prob-bar"
          role="progressbar"
          aria-valuenow={pct}
          aria-valuemin={0}
          aria-valuemax={100}
        >
          <div
            className="prob-fill"
            style={{ width: `${pct}%`, background: c.fill }}
          />
        </div>
        <span className="prob-label" style={{ color: c.fill }}>
          {pct}% probability
        </span>
      </div>

      <p className="risk-panel__updated">
        Updated: {new Date(current.timestamp_utc).toLocaleString('en-GB', { timeZone: 'UTC' })} UTC
      </p>

      {/* Risk factors */}
      {factors && factors.factors.length > 0 && (
        <div className="risk-panel__factors">
          <h4>Why is risk {current.risk_level.toLowerCase()}?</h4>
          <ul>
            {factors.factors.map((f) => (
              <li key={f.feature}>
                <span className="factor-label">
                  {FACTOR_LABELS[f.feature] ?? f.feature.replace(/_/g, ' ')}
                </span>
                {f.importance != null && (
                  <span className="factor-bar-wrapper">
                    <span
                      className="factor-bar"
                      style={{ width: `${Math.round(f.importance * 100)}%`, background: c.fill }}
                    />
                  </span>
                )}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* History chart */}
      {history && history.history.length > 1 && (
        <HistoryChart history={history.history} locationId={locationId} />
      )}
    </div>
  );
}
