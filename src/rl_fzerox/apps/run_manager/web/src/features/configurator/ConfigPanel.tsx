// src/rl_fzerox/apps/run_manager/web/src/features/configurator/ConfigPanel.tsx
import type { ReactNode } from "react";

import { IconButton } from "@/shared/ui/Button";
import { cn } from "@/shared/ui/cn";
import { ResetIcon } from "@/shared/ui/icons";

export function ConfigPanel({
  children,
  className,
  onReset,
  title,
  wide = false,
}: {
  children: ReactNode;
  className?: string;
  onReset?: () => void;
  title: string;
  wide?: boolean;
}) {
  return (
    <section
      className={cn(
        "grid content-start items-start gap-3 border border-app-border bg-app-surface-muted p-3.5",
        wide ? "col-span-full" : undefined,
        className,
      )}
    >
      <div className="flex items-center justify-between border-b border-app-border pb-2.5">
        <h3 className="m-0 text-[15px] font-bold text-app-text">{title}</h3>
        {onReset !== undefined ? (
          <IconButton
            aria-label={`Reset ${title} defaults`}
            className="tooltip-anchor"
            data-tooltip="Reset section defaults"
            size="small"
            tone="muted"
            onClick={onReset}
          >
            <ResetIcon />
          </IconButton>
        ) : null}
      </div>
      {children}
    </section>
  );
}
