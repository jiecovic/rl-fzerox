// src/rl_fzerox/apps/run_manager/web/src/shared/ui/icons/navigation.tsx
import type { IconProps } from "@/shared/ui/icons/types";

export function CloseIcon({ size = 14 }: IconProps) {
  return (
    <svg aria-hidden="true" fill="none" height={size} viewBox="0 0 16 16" width={size}>
      <path
        d="M4.25 4.25 8 8m0 0 3.75 3.75M8 8l3.75-3.75M8 8l-3.75 3.75"
        stroke="currentColor"
        strokeLinecap="round"
        strokeWidth="1.8"
      />
    </svg>
  );
}

export function ExpandAllIcon({ size = 16 }: IconProps) {
  return (
    <svg aria-hidden="true" fill="none" height={size} viewBox="0 0 20 20" width={size}>
      <path
        d="M5 8l5-5 5 5M5 12l5 5 5-5"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="1.7"
      />
    </svg>
  );
}

export function CollapseAllIcon({ size = 16 }: IconProps) {
  return (
    <svg aria-hidden="true" fill="none" height={size} viewBox="0 0 20 20" width={size}>
      <path
        d="M3 5l5 5-5 5M17 5l-5 5 5 5"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="1.7"
      />
    </svg>
  );
}

export function ArrowUpIcon({ size = 16 }: IconProps) {
  return (
    <svg aria-hidden="true" fill="none" height={size} viewBox="0 0 20 20" width={size}>
      <path
        d="M10 16V4M5 9l5-5 5 5"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="1.8"
      />
    </svg>
  );
}

export function ArrowDownIcon({ size = 16 }: IconProps) {
  return (
    <svg aria-hidden="true" fill="none" height={size} viewBox="0 0 20 20" width={size}>
      <path
        d="M10 4v12M5 11l5 5 5-5"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="1.8"
      />
    </svg>
  );
}

export function ChevronIcon({ size = 14 }: IconProps) {
  return (
    <svg aria-hidden="true" fill="none" height={size} viewBox="0 0 20 20" width={size}>
      <path
        d="M7 5.5 12 10 7 14.5"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="1.7"
      />
    </svg>
  );
}
