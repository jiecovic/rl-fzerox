// src/rl_fzerox/apps/run_manager/web/src/entities/runConfig/ui/sections/tracks/trackAssetUrls.ts
export function courseMinimapUrl(courseId: string) {
  return `/api/assets/course-minimaps/${encodeURIComponent(courseId)}`;
}

export function cupBannerUrl(cupId: string) {
  return `/api/assets/cup-banners/${encodeURIComponent(cupId)}`;
}
