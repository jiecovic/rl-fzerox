// src/rl_fzerox/apps/run_manager/web/src/features/configurator/ConfigDisclosure.tsx
import type { ReactNode } from "react";

import { ResetIcon } from "@/shared/ui/icons";
import { TooltipIconButton } from "@/shared/ui/TooltipIconButton";

export function ConfigDisclosure({
  children,
  defaultOpen = true,
  open,
  onReset,
  onToggle,
  title,
}: {
  children: ReactNode;
  defaultOpen?: boolean;
  open?: boolean;
  onReset?: () => void;
  onToggle?: (open: boolean) => void;
  title: string;
}) {
  return (
    <details
      className="group border border-app-border bg-app-surface"
      open={open ?? defaultOpen}
      onToggle={(event) => onToggle?.(event.currentTarget.open)}
    >
      <summary className="grid min-h-12 cursor-pointer list-none grid-cols-[minmax(0,1fr)_auto] items-center gap-3 px-3 text-sm font-bold text-app-text [&::-webkit-details-marker]:hidden">
        <span className="flex min-w-0 items-center gap-3">
          <span
            aria-hidden="true"
            className="w-4 flex-none text-app-muted before:content-['▸'] group-open:before:content-['▾']"
          />
          <span className="grid min-w-0 gap-1">
            <strong>{title}</strong>
          </span>
        </span>
        {onReset !== undefined ? (
          <TooltipIconButton
            aria-label={`Reset ${title} defaults`}
            size="small"
            tone="muted"
            tooltip="Reset defaults"
            onClick={(event) => {
              event.preventDefault();
              event.stopPropagation();
              onReset();
            }}
          >
            <ResetIcon />
          </TooltipIconButton>
        ) : null}
      </summary>
      <div className="grid gap-2.5 border-t border-app-border p-3">{children}</div>
    </details>
  );
}
