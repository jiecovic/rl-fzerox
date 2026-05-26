// src/rl_fzerox/apps/run_manager/web/src/shared/ui/Tooltip.tsx
import * as TooltipPrimitive from "@radix-ui/react-tooltip";
import type { ReactNode } from "react";

interface AppTooltipProps {
  children: ReactNode;
  content: string;
  side?: "left" | "top";
}

export function AppTooltip({ children, content, side = "top" }: AppTooltipProps) {
  return (
    <TooltipPrimitive.Provider delayDuration={160} skipDelayDuration={100}>
      <TooltipPrimitive.Root>
        <TooltipPrimitive.Trigger asChild>{children}</TooltipPrimitive.Trigger>
        <TooltipPrimitive.Portal>
          <TooltipPrimitive.Content
            className="z-50 max-w-[min(240px,calc(100vw-32px))] rounded-md border border-app-border-strong bg-app-surface px-2.5 py-2 text-xs leading-snug text-app-text shadow-[0_8px_20px_rgba(0,0,0,0.22)]"
            side={side}
            sideOffset={8}
          >
            {content}
            <TooltipPrimitive.Arrow className="fill-app-surface" />
          </TooltipPrimitive.Content>
        </TooltipPrimitive.Portal>
      </TooltipPrimitive.Root>
    </TooltipPrimitive.Provider>
  );
}
