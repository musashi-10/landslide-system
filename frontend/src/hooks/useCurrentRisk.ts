import { useState, useEffect } from 'react';
import { getCurrentRisk, getRiskHistory, getRiskFactors } from '../services/api';
import type {
  RiskCurrentResponse,
  RiskHistoryResponse,
  RiskFactorsResponse,
} from '../types/risk';

export function useCurrentRisk(locationId: string | null) {
  const [current, setCurrent] = useState<RiskCurrentResponse | null>(null);
  const [history, setHistory] = useState<RiskHistoryResponse | null>(null);
  const [factors, setFactors] = useState<RiskFactorsResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!locationId) {
      setCurrent(null);
      setHistory(null);
      setFactors(null);
      return;
    }

    let cancelled = false;
    setLoading(true);
    setError(null);

    Promise.all([
      getCurrentRisk(locationId),
      getRiskHistory(locationId),
      getRiskFactors(locationId),
    ])
      .then(([c, h, f]) => {
        if (!cancelled) {
          setCurrent(c);
          setHistory(h);
          setFactors(f);
        }
      })
      .catch((e) => {
        if (!cancelled) setError(e instanceof Error ? e.message : 'Failed to load risk data');
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => { cancelled = true; };
  }, [locationId]);

  return { current, history, factors, loading, error };
}
