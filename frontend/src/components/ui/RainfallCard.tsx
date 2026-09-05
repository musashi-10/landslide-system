import { CloudRain, CloudDrizzle, CloudLightning, CloudSnow } from 'lucide-react';
import type { RainfallData } from '../../types';

interface Props {
  data: RainfallData;
}

const LABELS: { key: keyof RainfallData; label: string; icon: typeof CloudRain }[] = [
  { key: 'rainfall_1h', label: '1h', icon: CloudDrizzle },
  { key: 'rainfall_6h', label: '6h', icon: CloudRain },
  { key: 'rainfall_24h', label: '24h', icon: CloudRain },
  { key: 'rainfall_3d', label: '3 Day', icon: CloudLightning },
  { key: 'rainfall_7d', label: '7 Day', icon: CloudSnow },
  { key: 'forecast_rainfall', label: 'Forecast', icon: CloudRain },
];

function getIntensityClass(mm: number | null): string {
  if (mm === null) return '';
  if (mm > 80) return 'rainfall--heavy';
  if (mm > 40) return 'rainfall--moderate';
  return 'rainfall--light';
}

export function RainfallCard({ data }: Props) {
  const hasAny = Object.values(data).some((v) => v !== null);
  if (!hasAny) return null;

  return (
    <div className="rainfall-card">
      <h4 className="rainfall-card__title">
        <CloudRain size={16} />
        Rainfall Conditions
      </h4>
      <div className="rainfall-card__grid">
        {LABELS.map(({ key, label, icon: Icon }) => {
          const val = data[key];
          return (
            <div key={key} className={`rainfall-cell ${getIntensityClass(val)}`}>
              <Icon size={14} className="rainfall-cell__icon" />
              <span className="rainfall-cell__label">{label}</span>
              <span className="rainfall-cell__value">
                {val !== null ? `${val.toFixed(1)} mm` : '—'}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
