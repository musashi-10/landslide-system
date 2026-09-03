import { RISK_COLORS } from '../../utils/riskColors';
import type { RiskLevel } from '../../types/risk';

const LEVELS: RiskLevel[] = ['LOW', 'MODERATE', 'HIGH', 'CRITICAL'];
const LABELS: Record<RiskLevel, string> = {
  LOW: 'Low Risk',
  MODERATE: 'Moderate Risk',
  HIGH: 'High Risk',
  CRITICAL: 'Critical Risk',
};

export function Legend() {
  return (
    <div className="legend">
      <h4 className="legend-title">Risk Level</h4>
      {LEVELS.map((level) => {
        const c = RISK_COLORS[level];
        return (
          <div key={level} className="legend-item">
            <span
              className="legend-dot"
              style={{ background: c.fill, border: `2px solid ${c.border}` }}
            />
            <span className="legend-label" style={{ color: '#e2e8f0' }}>
              {LABELS[level]}
            </span>
          </div>
        );
      })}
    </div>
  );
}
