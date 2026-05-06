// src/rl_fzerox/apps/run_manager/web/src/features/configurator/sections/tracks/cup_banners/index.ts
import jackBannerUrl from "./jack.svg";
import jokerBannerUrl from "./joker.svg";
import kingBannerUrl from "./king.svg";
import queenBannerUrl from "./queen.svg";
import xBannerUrl from "./x.svg";

export const CUP_BANNER_ASSETS = {
  jack: jackBannerUrl,
  joker: jokerBannerUrl,
  king: kingBannerUrl,
  queen: queenBannerUrl,
  x: xBannerUrl,
} as const;

export type CupBannerId = keyof typeof CUP_BANNER_ASSETS;
