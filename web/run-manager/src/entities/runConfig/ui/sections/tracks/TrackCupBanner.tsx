// web/run-manager/src/entities/runConfig/ui/sections/tracks/TrackCupBanner.tsx

import {
  CUP_BANNER_ASSETS,
  type CupBannerId,
} from "@/entities/runConfig/ui/sections/tracks/cup_banners";
import { cn } from "@/shared/ui/cn";

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
  if (!isCupBannerId(cupId)) {
    return (
      <div
        aria-hidden="true"
        className={cn(
          "track-cup-banner-fallback grid flex-none place-items-center border border-[color-mix(in_srgb,var(--cup-accent)_44%,var(--border))] bg-[color-mix(in_srgb,var(--cup-accent)_12%,var(--surface))] px-2.5 text-app-text",
          large ? "h-[62px] w-[45px]" : "h-11 w-8",
        )}
      >
        <span className="text-[11px] font-bold tracking-[0.08em] uppercase">
          {fallbackLabel(label)}
        </span>
      </div>
    );
  }
  return (
    <img
      alt=""
      aria-hidden="true"
      className={cn("block flex-none", large ? "h-[62px] w-[45px]" : "h-11 w-8")}
      draggable={false}
      src={CUP_BANNER_ASSETS[cupId]}
    />
  );
}
