// web/run-manager/src/shared/ui/config/ConfigPanel.tsx
import type { ReactNode } from "react";

import { cn } from "@/shared/ui/cn";
import { ResetIcon } from "@/shared/ui/icons";
import { TooltipIconButton } from "@/shared/ui/TooltipIconButton";

export function ConfigPanel({
  children,
  className,
  id,
  onReset,
  title,
  wide = false,
}: {
  children: ReactNode;
  className?: string;
  id?: string;
  onReset?: () => void;
  title: string;
  wide?: boolean;
}) {
  return (
    <section
      id={id}
      className={cn(
        "grid content-start items-start gap-3 border border-app-border bg-app-surface-muted p-3.5",
        wide ? "col-span-full" : undefined,
        className,
      )}
    >
      <div className="flex items-center justify-between border-b border-app-border pb-2.5">
        <h3 className="m-0 text-[15px] font-bold text-app-text">{title}</h3>
        {onReset !== undefined ? (
          <TooltipIconButton
            aria-label={`Reset ${title} defaults`}
            size="small"
            tone="muted"
            tooltip="Reset section defaults"
            onClick={onReset}
          >
            <ResetIcon />
          </TooltipIconButton>
        ) : null}
      </div>
      {children}
    </section>
  );
}
