import type { AlertEntry } from '../../types/risk';
import { RISK_COLORS } from '../../utils/riskColors';

interface Props {
  alerts: AlertEntry[];
}

export function AlertBanner({ alerts }: Props) {
  if (alerts.length === 0) return null;

  const topAlert = alerts[0];
  const color = RISK_COLORS[topAlert.risk_level];

  return (
    <div
      className="alert-banner"
      style={{ background: color.bg, borderColor: color.border }}
      role="alert"
      aria-live="assertive"
    >
      <span className="alert-icon">⚠️</span>
      <div className="alert-content">
        <strong>{topAlert.risk_level} RISK — ACTIVE WARNING</strong>
        {alerts.length > 1 && (
          <span className="alert-count"> (+{alerts.length - 1} more)</span>
        )}
        <p className="alert-message">{topAlert.message}</p>
      </div>
    </div>
  );
}
