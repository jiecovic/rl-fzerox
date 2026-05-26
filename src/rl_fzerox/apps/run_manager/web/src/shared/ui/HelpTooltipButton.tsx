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
      className="field-help inline-grid h-4 w-4 cursor-help place-items-center rounded-full border border-app-border bg-transparent p-0 text-[11px] leading-none text-app-muted hover:border-app-border-strong hover:text-app-text focus-visible:border-app-border-strong focus-visible:text-app-text"
      data-tooltip={text}
      data-tooltip-position={position}
      type="button"
    >
      <HelpIcon />
    </button>
  );
}
