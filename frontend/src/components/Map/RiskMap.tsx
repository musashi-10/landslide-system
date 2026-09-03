// @ts-nocheck  -- react-leaflet v4 has known prop-type gaps; suppressed for prototype
import { MapContainer, TileLayer, CircleMarker, Tooltip } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import type { RiskMapLocation } from '../../types/risk';
import { RISK_COLORS } from '../../utils/riskColors';

interface Props {
  locations: RiskMapLocation[];
  selectedId: string | null;
  onSelectLocation: (id: string) => void;
  loading: boolean;
  error: string | null;
  lastUpdated: Date | null;
}

const DARJEELING_CENTER = [27.4, 88.5] as [number, number];

export function RiskMap({
  locations,
  selectedId,
  onSelectLocation,
  loading,
  error,
  lastUpdated,
}: Props) {
  return (
    <div className="map-wrapper">
      {loading && <div className="map-overlay">Loading risk data…</div>}
      {error && (
        <div className="map-overlay map-error">
          <span>⚠️ Risk data temporarily unavailable</span>
          {lastUpdated && (
            <small>Last update: {lastUpdated.toLocaleTimeString()}</small>
          )}
        </div>
      )}
      <MapContainer
        center={DARJEELING_CENTER}
        zoom={9}
        className="leaflet-map"
        id="risk-map"
      >
        <TileLayer
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
        />
        {locations.map((loc) => {
          const c = RISK_COLORS[loc.risk_level];
          const isSelected = loc.location_id === selectedId;
          return (
            <CircleMarker
              key={loc.location_id}
              center={[loc.latitude, loc.longitude]}
              radius={isSelected ? 18 : 12}
              pathOptions={{
                fillColor: c.fill,
                fillOpacity: 0.85,
                color: isSelected ? '#fff' : c.border,
                weight: isSelected ? 3 : 1.5,
              }}
              eventHandlers={{ click: () => onSelectLocation(loc.location_id) }}
            >
              <Tooltip>
                <strong>{loc.location_id}</strong>
                <br />
                {loc.risk_level} — {(loc.risk_probability * 100).toFixed(0)}%
              </Tooltip>
            </CircleMarker>
          );
        })}
      </MapContainer>
    </div>
  );
}
