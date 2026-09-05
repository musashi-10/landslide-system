/**
 * Authority Risk Map page — full-screen map with filtering for critical-zone management.
 *
 * Wraps the existing RiskMap component with authority-specific controls:
 * risk level filtering, location detail panel, and critical-zone mode.
 */

import { useState } from 'react';
import { useRiskMap } from '../../hooks/useRiskMap';
import { useCurrentRisk } from '../../hooks/useCurrentRisk';
import { RiskMap } from '../../components/Map/RiskMap';
import { Legend } from '../../components/Legend/Legend';
import { RiskBadge } from '../../components/ui/RiskBadge';
import { RiskFactorList } from '../../components/ui/RiskFactorList';
import { RainfallCard } from '../../components/ui/RainfallCard';
import { LoadingState } from '../../components/ui/LoadingState';
import { extractRainfall } from '../../services/riskService';
import { RISK_COLORS } from '../../utils/riskColors';
import type { RiskLevel } from '../../types';
import {
  Filter,
  X,
  ShieldAlert,
  MapPin,
  Clock,
  RefreshCw,
} from 'lucide-react';

const LEVEL_OPTIONS: { value: string; label: string }[] = [
  { value: '', label: 'All Levels' },
  { value: 'CRITICAL', label: 'Critical Only' },
  { value: 'HIGH', label: 'High & Above' },
  { value: 'MODERATE', label: 'Moderate & Above' },
];

export function AuthRiskMap() {
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [levelFilter, setLevelFilter] = useState<string>('');
  const [criticalMode, setCriticalMode] = useState(false);

  const { data: mapData, loading: mapLoading, error: mapError, lastUpdated, reload } = useRiskMap(30_000);
  const { current, factors, loading: panelLoading } = useCurrentRisk(selectedId);

  const allLocations = mapData?.locations ?? [];

  // Apply filter
  const LEVEL_ORDER: RiskLevel[] = ['LOW', 'MODERATE', 'HIGH', 'CRITICAL'];
  const filteredLocations = allLocations.filter((loc) => {
    const effectiveFilter = criticalMode ? 'HIGH' : levelFilter;
    if (!effectiveFilter) return true;
    return LEVEL_ORDER.indexOf(loc.risk_level) >= LEVEL_ORDER.indexOf(effectiveFilter as RiskLevel);
  });

  const rainfall = factors ? extractRainfall(factors) : null;
  const pct = current ? Math.round(current.risk_probability * 100) : null;
  const riskColor = current ? RISK_COLORS[current.risk_level] : null;

  return (
    <div className="map-page">
      {/* ── Map Controls ───────────────────────────────────────────── */}
      <div className="map-controls">
        <div className="map-controls__left">
          <h1 className="map-controls__title">
            <MapPin size={18} /> Regional Risk Map
          </h1>
          {lastUpdated && (
            <span className="map-controls__updated">
              <Clock size={12} /> {lastUpdated.toLocaleTimeString('en-GB', { timeZone: 'UTC' })} UTC
            </span>
          )}
        </div>
        <div className="map-controls__right">
          <button
            className={`btn btn--sm ${criticalMode ? 'btn--danger' : 'btn--ghost'}`}
            onClick={() => { setCriticalMode((v) => !v); setLevelFilter(''); }}
          >
            <ShieldAlert size={14} />
            {criticalMode ? 'Exit Critical Mode' : 'Critical Zone Mode'}
          </button>

          <div className="filter-select-wrap">
            <Filter size={14} />
            <select
              className="filter-select"
              value={levelFilter}
              onChange={(e) => { setLevelFilter(e.target.value); setCriticalMode(false); }}
              disabled={criticalMode}
            >
              {LEVEL_OPTIONS.map((o) => (
                <option key={o.value} value={o.value}>{o.label}</option>
              ))}
            </select>
          </div>

          <button className="btn btn--ghost btn--sm" onClick={reload} title="Refresh map">
            <RefreshCw size={14} />
          </button>
        </div>
      </div>

      {/* ── Map + Side Panel ───────────────────────────────────────── */}
      <div className="map-page__body">
        {/* Map */}
        <section className="map-section" aria-label="Risk map">
          <RiskMap
            locations={filteredLocations}
            selectedId={selectedId}
            onSelectLocation={setSelectedId}
            loading={mapLoading}
            error={mapError}
            lastUpdated={lastUpdated}
          />
          <Legend />
          {criticalMode && (
            <div className="map-mode-badge">
              <ShieldAlert size={14} /> CRITICAL ZONE MODE — HIGH &amp; CRITICAL only
            </div>
          )}
        </section>

        {/* Location Detail Panel */}
        <aside className="map-detail-panel" aria-label="Location details">
          {!selectedId ? (
            <div className="map-detail-panel__empty">
              <MapPin size={32} />
              <p>Select a location on the map to view details</p>
            </div>
          ) : panelLoading ? (
            <LoadingState message="Loading location…" />
          ) : current ? (
            <div className="location-detail">
              <div className="location-detail__header">
                <div>
                  <h2 className="location-detail__id">{current.location_id}</h2>
                  <p className="location-detail__coords">
                    {allLocations.find(l => l.location_id === selectedId)?.latitude?.toFixed(3)},{' '}
                    {allLocations.find(l => l.location_id === selectedId)?.longitude?.toFixed(3)}
                  </p>
                </div>
                <button className="btn btn--ghost btn--sm" onClick={() => setSelectedId(null)}>
                  <X size={16} />
                </button>
              </div>

              <div className="location-detail__risk">
                <RiskBadge level={current.risk_level} size="lg" />
                <div className="location-detail__prob">
                  <span style={{ color: riskColor?.fill }}>{pct}%</span>
                  <span className="location-detail__prob-label">probability</span>
                </div>
              </div>

              <div className="prob-bar">
                <div className="prob-fill" style={{ width: `${pct}%`, background: riskColor?.fill }} />
              </div>

              <p className="location-detail__updated">
                <Clock size={12} />
                {new Date(current.timestamp_utc).toLocaleString('en-GB', { timeZone: 'UTC' })} UTC
              </p>

              {factors && factors.factors.length > 0 && (
                <RiskFactorList
                  factors={factors.factors}
                  riskLevel={current.risk_level}
                  title="Key Risk Factors"
                />
              )}

              {rainfall && <RainfallCard data={rainfall} />}
            </div>
          ) : null}
        </aside>
      </div>
    </div>
  );
}
