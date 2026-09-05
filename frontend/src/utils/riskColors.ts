import type { RiskLevel } from '../types/risk';

export const RISK_COLORS: Record<RiskLevel, { bg: string; border: string; text: string; fill: string }> = {
  LOW:      { bg: 'rgba(34,197,94,0.12)',   border: '#16a34a', text: '#86efac', fill: '#22c55e' },
  MODERATE: { bg: 'rgba(245,158,11,0.12)',  border: '#d97706', text: '#fde68a', fill: '#f59e0b' },
  HIGH:     { bg: 'rgba(239,68,68,0.12)',   border: '#dc2626', text: '#fca5a5', fill: '#ef4444' },
  CRITICAL: { bg: 'rgba(168,85,247,0.12)',  border: '#9333ea', text: '#d8b4fe', fill: '#a855f7' },
};

export function riskBadgeStyle(level: RiskLevel) {
  const c = RISK_COLORS[level];
  return {
    background: c.bg,
    border: `2px solid ${c.border}`,
    color: c.text,
  };
}
