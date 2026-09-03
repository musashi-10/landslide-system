import { useState, useEffect } from 'react';
import { getAlerts } from '../services/api';
import type { AlertEntry } from '../types/risk';

export function useAlerts(refreshMs = 15_000) {
  const [alerts, setAlerts] = useState<AlertEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = async () => {
    try {
      const result = await getAlerts();
      setAlerts(result.alerts.filter((a) => a.status === 'ACTIVE'));
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load alerts');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
    const interval = setInterval(load, refreshMs);
    return () => clearInterval(interval);
  }, [refreshMs]);

  return { alerts, loading, error };
}
