import { useState } from 'react';
import { useRiskMap } from './hooks/useRiskMap';
import { useCurrentRisk } from './hooks/useCurrentRisk';
import { useAlerts } from './hooks/useAlerts';
import { RiskMap } from './components/Map/RiskMap';
import { RiskPanel } from './components/RiskPanel/RiskPanel';
import { AlertBanner } from './components/AlertBanner/AlertBanner';
import { Legend } from './components/Legend/Legend';

const IS_MOCK = import.meta.env.VITE_USE_MOCK_API === 'true';

export default function App() {
  const [selectedId, setSelectedId] = useState<string | null>(null);

  const { data: mapData, loading: mapLoading, error: mapError, lastUpdated } = useRiskMap(30_000);
  const { current, history, factors, loading: panelLoading, error: panelError } = useCurrentRisk(selectedId);
  const { alerts } = useAlerts(15_000);

  const locations = mapData?.locations ?? [];

  return (
    <div className="app-shell">
      {/* ── Top Bar ──────────────────────────────────────────────────── */}
      <header className="topbar" role="banner">
        <div className="topbar__brand">
          <span className="topbar__icon" aria-hidden="true">🗺️</span>
          <h1 className="topbar__title">Landslide Early Warning System</h1>
        </div>
        <div className="topbar__meta">
          {IS_MOCK && <span className="mock-badge">MOCK DATA</span>}
          {lastUpdated && (
            <span className="last-updated" aria-live="polite">
              Updated {lastUpdated.toLocaleTimeString()}
            </span>
          )}
        </div>
      </header>

      {/* ── Alert Banner ──────────────────────────────────────────────── */}
      {alerts.length > 0 && <AlertBanner alerts={alerts} />}

      {/* ── Main Layout ──────────────────────────────────────────────── */}
      <main className="main-layout" role="main">
        {/* Left panel — risk detail */}
        <aside className="side-panel" aria-label="Location risk details">
          <RiskPanel
            locationId={selectedId ?? ''}
            current={current}
            history={history}
            factors={factors}
            loading={panelLoading}
            error={panelError}
          />
        </aside>

        {/* Map */}
        <section className="map-section" aria-label="Risk map">
          <RiskMap
            locations={locations}
            selectedId={selectedId}
            onSelectLocation={setSelectedId}
            loading={mapLoading}
            error={mapError}
            lastUpdated={lastUpdated}
          />
          <Legend />
        </section>
      </main>
    </div>
  );
}
