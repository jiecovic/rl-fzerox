// web/run-manager/src/shared/ui/Tooltip.tsx
import * as TooltipPrimitive from "@radix-ui/react-tooltip";
import type { ReactNode } from "react";

export type AppTooltipSide = "bottom" | "left" | "right" | "top";

interface AppTooltipProps {
  children: ReactNode;
  content: ReactNode;
  side?: AppTooltipSide;
}

interface OptionalAppTooltipProps extends Omit<AppTooltipProps, "content"> {
  content?: ReactNode;
}

export function AppTooltipProvider({ children }: { children: ReactNode }) {
  return (
    <TooltipPrimitive.Provider delayDuration={0} skipDelayDuration={250}>
      {children}
    </TooltipPrimitive.Provider>
  );
}

export function AppTooltip({ children, content, side = "top" }: AppTooltipProps) {
  return (
    <TooltipPrimitive.Root>
      <TooltipPrimitive.Trigger asChild>{children}</TooltipPrimitive.Trigger>
      <TooltipPrimitive.Portal>
        <TooltipPrimitive.Content
          className="z-50 max-w-[min(240px,calc(100vw-32px))] rounded-md border border-app-border-strong bg-app-surface px-2.5 py-2 text-xs leading-snug text-app-text shadow-[0_8px_20px_rgba(0,0,0,0.22)]"
          side={side}
          sideOffset={8}
        >
          {content}
          <TooltipPrimitive.Arrow className="fill-app-surface [filter:drop-shadow(0_1px_0_var(--border-strong))] [stroke-width:1px] [stroke:var(--border-strong)]" />
        </TooltipPrimitive.Content>
      </TooltipPrimitive.Portal>
    </TooltipPrimitive.Root>
  );
}

export function OptionalAppTooltip({ children, content, side }: OptionalAppTooltipProps) {
  if (content === undefined || content === null || content === "") {
    return <>{children}</>;
  }
  return (
    <AppTooltip content={content} side={side}>
      {children}
    </AppTooltip>
  );
}
