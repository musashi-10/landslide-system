interface Props {
  message?: string;
  size?: 'sm' | 'md' | 'lg';
}

export function LoadingState({ message = 'Loading…', size = 'md' }: Props) {
  return (
    <div className={`loading-state loading-state--${size}`}>
      <div className="spinner" />
      <p className="loading-state__msg">{message}</p>
    </div>
  );
}

export function SkeletonLine({ width = '100%' }: { width?: string }) {
  return <div className="skeleton-line" style={{ width }} />;
}

export function SkeletonCard() {
  return (
    <div className="skeleton-card">
      <SkeletonLine width="60%" />
      <SkeletonLine width="40%" />
      <SkeletonLine width="80%" />
    </div>
  );
}
