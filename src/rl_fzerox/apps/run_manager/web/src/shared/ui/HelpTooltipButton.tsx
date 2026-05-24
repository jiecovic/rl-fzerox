// src/rl_fzerox/apps/run_manager/web/src/shared/ui/HelpTooltipButton.tsx
import { HelpIcon } from "@/shared/ui/icons";

export function HelpTooltipButton({
  label,
  position,
  text,
}: {
  label: string;
  position?: "left";
  text: string;
}) {
  return (
    <button
      aria-label={`${label}: ${text}`}
      className="field-help"
      data-tooltip={text}
      data-tooltip-position={position}
      type="button"
    >
      <HelpIcon />
    </button>
  );
}
