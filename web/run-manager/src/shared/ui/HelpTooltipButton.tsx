// web/run-manager/src/shared/ui/HelpTooltipButton.tsx
import { HelpIcon } from "@/shared/ui/icons";
import { AppTooltip } from "@/shared/ui/Tooltip";

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
    <AppTooltip content={text} side={position === "left" ? "left" : "top"}>
      <button
        aria-label={`${label}: ${text}`}
        className="inline-grid h-4 w-4 cursor-help place-items-center rounded-full border border-app-border bg-transparent p-0 text-[11px] leading-none text-app-muted hover:border-app-border-strong hover:text-app-text focus-visible:border-app-border-strong focus-visible:text-app-text"
        type="button"
      >
        <HelpIcon />
      </button>
    </AppTooltip>
  );
}
