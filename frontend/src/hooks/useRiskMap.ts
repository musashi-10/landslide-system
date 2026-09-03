import { useState, useEffect } from 'react';
import { getRiskMap } from '../services/api';
import type { RiskMapResponse } from '../types/risk';

export function useRiskMap(refreshMs = 30_000) {
  const [data, setData] = useState<RiskMapResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);

  const load = async () => {
    try {
      const result = await getRiskMap();
      setData(result);
      setLastUpdated(new Date());
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load risk map');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
    const interval = setInterval(load, refreshMs);
    return () => clearInterval(interval);
  }, [refreshMs]);

  return { data, loading, error, lastUpdated, reload: load };
}
