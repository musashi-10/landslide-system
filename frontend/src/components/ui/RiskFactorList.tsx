import { TrendingUp } from 'lucide-react';
import type { RiskFactor } from '../../types';
import { RISK_COLORS } from '../../utils/riskColors';
import type { RiskLevel } from '../../types';

interface Props {
  factors: RiskFactor[];
  riskLevel: RiskLevel;
  title?: string;
}

const FACTOR_LABELS: Record<string, string> = {
  rainfall_24h: '🌧️ Heavy 24h rainfall',
  high_24h_rainfall: '🌧️ Heavy 24h rainfall',
  rainfall_7d: '🌧️ High 7-day rainfall',
  high_7d_rainfall: '🌧️ High 7-day rainfall',
  slope: '⛰️ Steep terrain slope',
  steep_slope: '⛰️ Steep terrain slope',
  high_susceptibility: '📍 High susceptibility index',
  historical_landslide_activity: '📋 Historical landslide activity',
  high_soil_moisture: '💧 High soil moisture',
  intense_1h_rainfall: '⛈️ Intense recent rainfall',
  rainfall_1h: '⛈️ Recent 1h rainfall',
  rainfall_6h: '🌧️ 6h rainfall',
  rainfall_3d: '🌧️ 3-day rainfall',
  forecast_rainfall: '🌦️ Rainfall forecast',
  moisture_indicator: '💧 Moisture indicator',
};

export function RiskFactorList({ factors, riskLevel, title = 'Why is risk elevated?' }: Props) {
  const color = RISK_COLORS[riskLevel];
  // Only show top risk factors (importance-based, non-rainfall for the factor list)
  const topFactors = factors
    .filter((f) => f.importance !== null && f.importance > 0.05)
    .sort((a, b) => (b.importance ?? 0) - (a.importance ?? 0))
    .slice(0, 5);

  if (topFactors.length === 0) return null;

  return (
    <div className="risk-factor-list">
      <h4 className="risk-factor-list__title">
        <TrendingUp size={14} />
        {title}
      </h4>
      <ul className="risk-factor-list__items">
        {topFactors.map((f) => (
          <li key={f.feature} className="risk-factor-item">
            <div className="risk-factor-item__header">
              <span className="risk-factor-item__label">
                {FACTOR_LABELS[f.feature] ?? f.feature.replace(/_/g, ' ')}
              </span>
              {f.importance !== null && (
                <span className="risk-factor-item__pct">
                  {Math.round(f.importance * 100)}%
                </span>
              )}
            </div>
            {f.importance !== null && (
              <div className="risk-factor-item__bar-bg">
                <div
                  className="risk-factor-item__bar-fill"
                  style={{
                    width: `${Math.round(f.importance * 100)}%`,
                    background: color.fill,
                  }}
                />
              </div>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}
