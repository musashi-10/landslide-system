import type { RiskLevel } from '../types/risk';

export const RISK_COLORS: Record<RiskLevel, { bg: string; border: string; text: string; fill: string }> = {
  LOW:      { bg: '#d1fae5', border: '#059669', text: '#064e3b', fill: '#10b981' },
  MODERATE: { bg: '#fef3c7', border: '#d97706', text: '#78350f', fill: '#f59e0b' },
  HIGH:     { bg: '#fee2e2', border: '#dc2626', text: '#7f1d1d', fill: '#ef4444' },
  CRITICAL: { bg: '#4c1d95', border: '#7c3aed', text: '#ede9fe', fill: '#8b5cf6' },
};

export function riskBadgeStyle(level: RiskLevel) {
  const c = RISK_COLORS[level];
  return {
    background: c.bg,
    border: `2px solid ${c.border}`,
    color: c.text,
  };
}
