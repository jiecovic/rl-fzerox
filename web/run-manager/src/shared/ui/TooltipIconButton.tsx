// web/run-manager/src/shared/ui/TooltipIconButton.tsx
import type { ComponentProps } from "react";

import { IconButton } from "@/shared/ui/Button";
import { AppTooltip, type AppTooltipSide } from "@/shared/ui/Tooltip";

interface TooltipIconButtonProps extends Omit<ComponentProps<typeof IconButton>, "title"> {
  side?: AppTooltipSide;
  tooltip: string;
}

export function TooltipIconButton({ children, side, tooltip, ...props }: TooltipIconButtonProps) {
  return (
    <AppTooltip content={tooltip} side={side}>
      <span className="inline-flex">
        <IconButton {...props}>{children}</IconButton>
      </span>
    </AppTooltip>
  );
}
