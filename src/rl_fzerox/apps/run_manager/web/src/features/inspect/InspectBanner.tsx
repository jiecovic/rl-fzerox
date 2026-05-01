interface InspectBannerProps {
  title: string;
  subtitle: string;
  onClose: () => void;
  onOpen: () => void;
}

export function InspectBanner({ title, subtitle, onClose, onOpen }: InspectBannerProps) {
  return (
    <div className="inspect-banner">
      <button className="inspect-banner-main" type="button" onClick={onOpen}>
        <strong>{title}</strong>
        <span>{subtitle}</span>
      </button>
      <button
        aria-label="Close inspector"
        className="icon-button tooltip-anchor"
        data-tooltip="Close inspector"
        type="button"
        onClick={onClose}
      >
        <CloseIcon />
      </button>
    </div>
  );
}

function CloseIcon() {
  return (
    <svg aria-hidden="true" fill="none" height="18" viewBox="0 0 20 20" width="18">
      <path
        d="m5.5 5.5 9 9M14.5 5.5l-9 9"
        stroke="currentColor"
        strokeLinecap="round"
        strokeWidth="1.8"
      />
    </svg>
  );
}
