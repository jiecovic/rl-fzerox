import {
  CUP_BANNER_ASSETS,
  type CupBannerId,
} from "@/features/configurator/sections/tracks/cup_banners";

interface TrackCupBannerProps {
  cupId: string;
  label: string;
  large?: boolean;
}

function fallbackLabel(label: string): string {
  return label.replace(/\s+cup$/i, "");
}

function isCupBannerId(cupId: string): cupId is CupBannerId {
  return cupId in CUP_BANNER_ASSETS;
}

export function TrackCupBanner({ cupId, label, large = false }: TrackCupBannerProps) {
  const className = large ? "track-cup-banner-image large" : "track-cup-banner-image";
  if (!isCupBannerId(cupId)) {
    return (
      <div
        aria-hidden="true"
        className={large ? "track-cup-banner-fallback large" : "track-cup-banner-fallback"}
      >
        <span>{fallbackLabel(label)}</span>
      </div>
    );
  }
  return (
    <img
      alt=""
      aria-hidden="true"
      className={className}
      draggable={false}
      src={CUP_BANNER_ASSETS[cupId]}
    />
  );
}
