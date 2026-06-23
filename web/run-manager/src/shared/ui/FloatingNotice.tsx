// web/run-manager/src/shared/ui/FloatingNotice.tsx
import type { ReactNode } from "react";

import { cn } from "@/shared/ui/cn";
import { CloseIcon } from "@/shared/ui/icons";
import { TooltipIconButton } from "@/shared/ui/TooltipIconButton";

interface FloatingNoticeProps {
  children: ReactNode;
  onDismiss?: () => void;
  tone?: "default" | "error";
}

export function FloatingNotice({ children, onDismiss, tone = "default" }: FloatingNoticeProps) {
  return (
    <div className="pointer-events-none fixed top-4 right-4 z-50 grid max-w-[min(520px,calc(100vw-2rem))] gap-2">
      <div
        className={cn(
          "pointer-events-auto flex items-start gap-3 border border-app-border bg-app-surface-muted px-4 py-3 text-sm shadow-[0_18px_42px_rgba(0,0,0,0.32)]",
          tone === "error" ? "border-app-danger text-app-danger" : "text-app-text",
        )}
        role={tone === "error" ? "alert" : "status"}
      >
        <div className="min-w-0 flex-1">{children}</div>
        {onDismiss !== undefined ? (
          <TooltipIconButton
            aria-label="Dismiss notice"
            className="mt-[-5px] mr-[-6px] shrink-0"
            size="small"
            tone={tone === "error" ? "danger" : "muted"}
            tooltip="Dismiss"
            onClick={onDismiss}
          >
            <CloseIcon />
          </TooltipIconButton>
        ) : null}
      </div>
    </div>
  );
}
