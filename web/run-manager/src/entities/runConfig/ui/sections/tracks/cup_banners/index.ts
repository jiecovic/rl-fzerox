// web/run-manager/src/entities/runConfig/ui/sections/tracks/cup_banners/index.ts
import jackBannerUrl from "@/entities/runConfig/ui/sections/tracks/cup_banners/jack.svg";
import jokerBannerUrl from "@/entities/runConfig/ui/sections/tracks/cup_banners/joker.svg";
import kingBannerUrl from "@/entities/runConfig/ui/sections/tracks/cup_banners/king.svg";
import queenBannerUrl from "@/entities/runConfig/ui/sections/tracks/cup_banners/queen.svg";
import xBannerUrl from "@/entities/runConfig/ui/sections/tracks/cup_banners/x.svg";

export const CUP_BANNER_ASSETS = {
  jack: jackBannerUrl,
  joker: jokerBannerUrl,
  king: kingBannerUrl,
  queen: queenBannerUrl,
  x: xBannerUrl,
} as const;

export type CupBannerId = keyof typeof CUP_BANNER_ASSETS;
