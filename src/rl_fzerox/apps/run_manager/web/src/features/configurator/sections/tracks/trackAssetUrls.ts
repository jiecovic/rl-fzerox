export function courseMinimapUrl(courseId: string) {
  return `/api/assets/course-minimaps/${encodeURIComponent(courseId)}`;
}

export function cupBannerUrl(cupId: string) {
  return `/api/assets/cup-banners/${encodeURIComponent(cupId)}`;
}
