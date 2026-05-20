// src/rl_fzerox/apps/run_manager/web/src/shared/ui/icons/actions.tsx
import type { IconProps } from "@/shared/ui/icons/types";

export function PlusIcon({ size = 14 }: IconProps) {
  return (
    <svg aria-hidden="true" fill="none" height={size} viewBox="0 0 20 20" width={size}>
      <path
        d="M10 4.5v11M4.5 10h11"
        stroke="currentColor"
        strokeLinecap="round"
        strokeWidth="1.7"
      />
    </svg>
  );
}

export function TrashIcon({ size = 14 }: IconProps) {
  return (
    <svg aria-hidden="true" fill="none" height={size} viewBox="0 0 20 20" width={size}>
      <path
        d="M6.5 6.5v7M10 6.5v7M13.5 6.5v7M4.5 4.5h11M7.5 4.5l.8-1.5h3.4l.8 1.5M6 4.5v11h8v-11"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="1.5"
      />
    </svg>
  );
}

export function RandomizeIcon({ size = 18 }: IconProps) {
  return (
    <svg aria-hidden="true" fill="none" height={size} viewBox="0 0 20 20" width={size}>
      <path
        d="M15.2 7.2a5.6 5.6 0 0 0-9.7-1.4M4.8 12.8a5.6 5.6 0 0 0 9.7 1.4"
        stroke="currentColor"
        strokeLinecap="round"
        strokeWidth="1.7"
      />
      <path
        d="M15.6 3.8v3.6h-3.6M4.4 16.2v-3.6h3.6"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="1.7"
      />
    </svg>
  );
}

export function SaveDraftIcon({ size = 14 }: IconProps) {
  return (
    <svg aria-hidden="true" fill="none" height={size} viewBox="0 0 20 20" width={size}>
      <path
        d="M4.5 3.5h8l3 3v10h-11zM7 3.5v5h6v-5M7 13h6"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="1.5"
      />
    </svg>
  );
}

export function ResetIcon({ size = 12 }: IconProps) {
  return (
    <svg aria-hidden="true" fill="none" height={size} viewBox="0 0 24 24" width={size}>
      <path
        d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="2"
      />
      <path
        d="M3 3v5h5"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="2"
      />
    </svg>
  );
}

export function AddLayerIcon({ size = 14 }: IconProps) {
  return (
    <svg aria-hidden="true" fill="none" height={size} viewBox="0 0 20 20" width={size}>
      <path d="M10 4v12M4 10h12" stroke="currentColor" strokeLinecap="round" strokeWidth="1.8" />
    </svg>
  );
}

export function RemoveLayerIcon({ size = 12 }: IconProps) {
  return (
    <svg aria-hidden="true" fill="none" height={size} viewBox="0 0 20 20" width={size}>
      <path d="M5 10h10" stroke="currentColor" strokeLinecap="round" strokeWidth="1.8" />
    </svg>
  );
}

export function CnnConfigIcon({ size = 14 }: IconProps) {
  return (
    <svg aria-hidden="true" fill="none" height={size} viewBox="0 0 20 20" width={size}>
      <rect height="12" rx="2" stroke="currentColor" strokeWidth="1.4" width="12" x="4" y="4" />
      <path
        d="M8 8h4M8 12h7M12 8v4"
        stroke="currentColor"
        strokeLinecap="round"
        strokeWidth="1.4"
      />
    </svg>
  );
}

export function ForkIcon({ size = 16 }: IconProps) {
  return (
    <svg aria-hidden="true" fill="none" height={size} viewBox="0 0 20 20" width={size}>
      <path
        d="M6 4a1.75 1.75 0 1 1-3.5 0A1.75 1.75 0 0 1 6 4Zm11.5 0a1.75 1.75 0 1 1-3.5 0a1.75 1.75 0 0 1 3.5 0ZM6 16a1.75 1.75 0 1 1-3.5 0A1.75 1.75 0 0 1 6 16ZM4.25 5.75v7.5M5.5 10h6.25c1.24 0 2.25-1.01 2.25-2.25V5.75"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="1.5"
      />
    </svg>
  );
}

export function FolderIcon({ size = 14 }: IconProps) {
  return (
    <svg aria-hidden="true" fill="none" height={size} viewBox="0 0 20 20" width={size}>
      <path
        d="M3.5 5.5h4.2l1.4 1.7h7.4v7.3H3.5z"
        stroke="currentColor"
        strokeLinejoin="round"
        strokeWidth="1.5"
      />
    </svg>
  );
}

export function ChartIcon({ size = 14 }: IconProps) {
  return (
    <svg aria-hidden="true" fill="none" height={size} viewBox="0 0 20 20" width={size}>
      <path
        d="M4 15.5V4.5M9 15.5v-6M14 15.5V7M3.5 15.5h13"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="1.5"
      />
    </svg>
  );
}

export function WatchIcon({ size = 14 }: IconProps) {
  return (
    <svg aria-hidden="true" fill="none" height={size} viewBox="0 0 20 20" width={size}>
      <path
        d="M10 5c-3.7 0-6.8 2-8 5 1.2 3 4.3 5 8 5s6.8-2 8-5c-1.2-3-4.3-5-8-5Z"
        stroke="currentColor"
        strokeLinejoin="round"
        strokeWidth="1.5"
      />
      <circle cx="10" cy="10" r="2.3" stroke="currentColor" strokeWidth="1.5" />
    </svg>
  );
}

export function StopIcon({ size = 14 }: IconProps) {
  return (
    <svg aria-hidden="true" fill="none" height={size} viewBox="0 0 20 20" width={size}>
      <rect
        height="9"
        rx="1.25"
        stroke="currentColor"
        strokeWidth="1.5"
        width="9"
        x="5.5"
        y="5.5"
      />
    </svg>
  );
}

export function ResumeIcon({ size = 14 }: IconProps) {
  return (
    <svg aria-hidden="true" fill="none" height={size} viewBox="0 0 20 20" width={size}>
      <path
        d="M7 5.5v9l7-4.5-7-4.5Z"
        stroke="currentColor"
        strokeLinejoin="round"
        strokeWidth="1.5"
      />
    </svg>
  );
}

export function ExportIcon({ size = 14 }: IconProps) {
  return (
    <svg aria-hidden="true" fill="none" height={size} viewBox="0 0 20 20" width={size}>
      <path
        d="M10 3.5v8M6.75 8.25 10 11.5l3.25-3.25M4.5 14.5v2h11v-2"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="1.5"
      />
    </svg>
  );
}

export function ImportIcon({ size = 14 }: IconProps) {
  return (
    <svg aria-hidden="true" fill="none" height={size} viewBox="0 0 20 20" width={size}>
      <path
        d="M10 16.5v-8M6.75 11.75 10 8.5l3.25 3.25M4.5 5.5v-2h11v2"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="1.5"
      />
    </svg>
  );
}

export function BranchSourceIcon({ size = 16 }: IconProps) {
  return (
    <svg aria-hidden="true" fill="none" height={size} viewBox="0 0 16 16" width={size}>
      <path
        d="M5 3.5a1.5 1.5 0 1 1-3 0A1.5 1.5 0 0 1 5 3.5Zm0 9a1.5 1.5 0 1 1-3 0A1.5 1.5 0 0 1 5 12.5Zm0-9v2c0 1.105.895 2 2 2h2.25a1.75 1.75 0 1 1 0 1H7a3 3 0 0 1-3-3v-2"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="1.25"
      />
    </svg>
  );
}

export function CopyIcon({ size = 14 }: IconProps) {
  return (
    <svg aria-hidden="true" fill="none" height={size} viewBox="0 0 20 20" width={size}>
      <rect x="7" y="4" width="9" height="11" rx="1.5" stroke="currentColor" strokeWidth="1.5" />
      <path
        d="M5 12.5H4.5A1.5 1.5 0 0 1 3 11V5.5A1.5 1.5 0 0 1 4.5 4H10"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="1.5"
      />
    </svg>
  );
}
