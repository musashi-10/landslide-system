import type { RiskLevel } from '../../types';
import { RISK_COLORS } from '../../utils/riskColors';

interface Props {
  level: RiskLevel;
  size?: 'sm' | 'md' | 'lg';
  showDot?: boolean;
}

export function RiskBadge({ level, size = 'md', showDot = true }: Props) {
  const c = RISK_COLORS[level];
  const sizeClass = `risk-badge--${size}`;

  return (
    <span
      className={`risk-badge ${sizeClass}`}
      style={{
        background: c.bg,
        border: `2px solid ${c.border}`,
        color: c.text,
      }}
    >
      {showDot && (
        <span
          className="risk-badge__dot"
          style={{ background: c.fill }}
        />
      )}
      {level}
    </span>
  );
}
