import type { ReactNode } from 'react';

interface Props {
  label: string;
  value: string | number;
  icon?: ReactNode;
  accentColor?: string;
  subtitle?: string;
  onClick?: () => void;
}

export function StatCard({ label, value, icon, accentColor, subtitle, onClick }: Props) {
  return (
    <div
      className={`stat-card ${onClick ? 'stat-card--clickable' : ''}`}
      style={accentColor ? { borderTopColor: accentColor } : undefined}
      onClick={onClick}
      role={onClick ? 'button' : undefined}
      tabIndex={onClick ? 0 : undefined}
      onKeyDown={onClick ? (e) => { if (e.key === 'Enter') onClick(); } : undefined}
    >
      {icon && <div className="stat-card__icon">{icon}</div>}
      <div className="stat-card__body">
        <span className="stat-card__value" style={accentColor ? { color: accentColor } : undefined}>
          {value}
        </span>
        <span className="stat-card__label">{label}</span>
        {subtitle && <span className="stat-card__subtitle">{subtitle}</span>}
      </div>
    </div>
  );
}
