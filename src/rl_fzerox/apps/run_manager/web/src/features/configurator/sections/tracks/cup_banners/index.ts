// src/rl_fzerox/apps/run_manager/web/src/features/configurator/sections/tracks/cup_banners/index.ts
import jackBannerUrl from "@/features/configurator/sections/tracks/cup_banners/jack.svg";
import jokerBannerUrl from "@/features/configurator/sections/tracks/cup_banners/joker.svg";
import kingBannerUrl from "@/features/configurator/sections/tracks/cup_banners/king.svg";
import queenBannerUrl from "@/features/configurator/sections/tracks/cup_banners/queen.svg";
import xBannerUrl from "@/features/configurator/sections/tracks/cup_banners/x.svg";

export const CUP_BANNER_ASSETS = {
  jack: jackBannerUrl,
  joker: jokerBannerUrl,
  king: kingBannerUrl,
  queen: queenBannerUrl,
  x: xBannerUrl,
} as const;

export type CupBannerId = keyof typeof CUP_BANNER_ASSETS;
