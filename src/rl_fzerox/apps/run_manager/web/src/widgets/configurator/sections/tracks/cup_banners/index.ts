// src/rl_fzerox/apps/run_manager/web/src/widgets/configurator/sections/tracks/cup_banners/index.ts
import jackBannerUrl from "@/widgets/configurator/sections/tracks/cup_banners/jack.svg";
import jokerBannerUrl from "@/widgets/configurator/sections/tracks/cup_banners/joker.svg";
import kingBannerUrl from "@/widgets/configurator/sections/tracks/cup_banners/king.svg";
import queenBannerUrl from "@/widgets/configurator/sections/tracks/cup_banners/queen.svg";
import xBannerUrl from "@/widgets/configurator/sections/tracks/cup_banners/x.svg";

export const CUP_BANNER_ASSETS = {
  jack: jackBannerUrl,
  joker: jokerBannerUrl,
  king: kingBannerUrl,
  queen: queenBannerUrl,
  x: xBannerUrl,
} as const;

export type CupBannerId = keyof typeof CUP_BANNER_ASSETS;
