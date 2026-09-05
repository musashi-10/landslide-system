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
      style={{
        background: color.bg,
        borderColor: color.border,
        color: color.text,
        display: 'flex',
        alignItems: 'flex-start',
        gap: '0.75rem',
        padding: '0.6rem 1.5rem',
        borderBottom: `2px solid ${color.border}`,
        flexShrink: 0,
        animation: 'slide-down 0.3s ease',
      }}
      role="alert"
      aria-live="assertive"
    >
      <span style={{ fontSize: '1.25rem', marginTop: 2 }}>⚠️</span>
      <div>
        <strong style={{ display: 'block', fontSize: '0.85rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
          {topAlert.risk_level} RISK — ACTIVE WARNING
          {alerts.length > 1 && <span style={{ fontWeight: 400, fontSize: '0.8rem' }}> (+{alerts.length - 1} more)</span>}
        </strong>
        <p style={{ fontSize: '0.8rem', opacity: 0.85, marginTop: 2 }}>{topAlert.message}</p>
      </div>
    </div>
  );
}
