import { AlertTriangle, RefreshCw } from 'lucide-react';

interface Props {
  title?: string;
  message: string;
  onRetry?: () => void;
  lastUpdated?: Date | null;
}

export function ErrorState({ title = 'Something went wrong', message, onRetry, lastUpdated }: Props) {
  return (
    <div className="error-state">
      <AlertTriangle className="error-state__icon" size={40} />
      <h3 className="error-state__title">{title}</h3>
      <p className="error-state__message">{message}</p>
      {lastUpdated && (
        <p className="error-state__timestamp">
          Last successful update: {lastUpdated.toLocaleTimeString('en-GB', { timeZone: 'UTC' })} UTC
        </p>
      )}
      {onRetry && (
        <button className="btn btn--secondary" onClick={onRetry}>
          <RefreshCw size={14} />
          Retry
        </button>
      )}
    </div>
  );
}
