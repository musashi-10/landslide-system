import {
  LineChart, Line, XAxis, YAxis, Tooltip,
  CartesianGrid, ResponsiveContainer, ReferenceLine,
} from 'recharts';
import type { RiskHistoryEntry } from '../../types/risk';

interface Props {
  history: RiskHistoryEntry[];
  locationId: string;
}

function formatTime(ts: string): string {
  try {
    const d = new Date(ts);
    return `${d.getMonth() + 1}/${d.getDate()} ${d.getHours()}:00`;
  } catch {
    return ts;
  }
}

export function HistoryChart({ history, locationId }: Props) {
  const data = history.map((h) => ({
    time: formatTime(h.timestamp_utc),
    probability: Math.round(h.risk_probability * 100),
    level: h.risk_level,
  }));

  return (
    <div className="history-chart">
      <h4>Risk History — {locationId}</h4>
      <ResponsiveContainer width="100%" height={160}>
        <LineChart data={data} margin={{ top: 4, right: 8, bottom: 4, left: -20 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
          <XAxis dataKey="time" tick={{ fontSize: 10, fill: '#94a3b8' }} />
          <YAxis domain={[0, 100]} tick={{ fontSize: 10, fill: '#94a3b8' }} unit="%" />
          <Tooltip
            contentStyle={{ background: '#1e293b', border: '1px solid #334155', color: '#e2e8f0' }}
          formatter={(v: unknown) => [`${Number(v)}%`, 'Probability']}
          />
          <ReferenceLine y={65} stroke="#ef4444" strokeDasharray="4 4" label={{ value: 'HIGH', fill: '#ef4444', fontSize: 10 }} />
          <ReferenceLine y={85} stroke="#8b5cf6" strokeDasharray="4 4" label={{ value: 'CRITICAL', fill: '#8b5cf6', fontSize: 10 }} />
          <Line
            type="monotone"
            dataKey="probability"
            stroke="#38bdf8"
            strokeWidth={2}
            dot={{ r: 3, fill: '#38bdf8' }}
            activeDot={{ r: 5 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
