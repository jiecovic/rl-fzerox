// src/rl_fzerox/apps/run_manager/web/src/features/configurator/sections/tracks/trackAssetUrls.ts
export function courseMinimapUrl(courseId: string) {
  return `/api/assets/course-minimaps/${encodeURIComponent(courseId)}`;
}

export function cupBannerUrl(cupId: string) {
  return `/api/assets/cup-banners/${encodeURIComponent(cupId)}`;
}
