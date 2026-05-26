// src/rl_fzerox/apps/run_manager/web/src/shared/ui/cn.ts
type ClassValue = false | null | string | undefined;

export function cn(...values: ClassValue[]): string {
  return values.filter(Boolean).join(" ");
}
