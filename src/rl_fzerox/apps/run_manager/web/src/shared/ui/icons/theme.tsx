// src/rl_fzerox/apps/run_manager/web/src/shared/ui/icons/theme.tsx
import type { IconProps } from "@/shared/ui/icons/types";

export function MoonIcon({ size = 20 }: IconProps) {
  return (
    <svg aria-hidden="true" fill="none" height={size} viewBox="0 0 24 24" width={size}>
      <path
        d="M21 14.4A7.7 7.7 0 0 1 9.6 3a8.8 8.8 0 1 0 11.4 11.4Z"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="1.8"
      />
    </svg>
  );
}

export function SunIcon({ size = 20 }: IconProps) {
  return (
    <svg aria-hidden="true" fill="none" height={size} viewBox="0 0 24 24" width={size}>
      <circle cx="12" cy="12" r="4.2" stroke="currentColor" strokeWidth="1.8" />
      <path
        d="M12 2.8v2.1M12 19.1v2.1M4.1 4.1l1.5 1.5M18.4 18.4l1.5 1.5M2.8 12h2.1M19.1 12h2.1M4.1 19.9l1.5-1.5M18.4 5.6l1.5-1.5"
        stroke="currentColor"
        strokeLinecap="round"
        strokeWidth="1.8"
      />
    </svg>
  );
}
