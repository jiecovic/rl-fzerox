// src/rl_fzerox/apps/run_manager/web/src/shared/ui/configFields/toggleSwitch.tsx
import { cn } from "@/shared/ui/cn";
import { OptionalAppTooltip } from "@/shared/ui/Tooltip";

interface ToggleSwitchProps {
  checked: boolean;
  disabled?: boolean;
  hideLabel?: boolean;
  label: string;
  tooltip?: string;
  onChange: (checked: boolean) => void;
}

export function ToggleSwitch({
  checked,
  disabled = false,
  hideLabel = false,
  label,
  tooltip,
  onChange,
}: ToggleSwitchProps) {
  const control = (
    <label
      className={cn(
        "inline-flex items-center gap-2 text-xs font-semibold text-app-muted",
        disabled ? "cursor-not-allowed opacity-50" : "cursor-pointer",
      )}
    >
      <input
        aria-label={label}
        className="peer sr-only"
        checked={checked}
        disabled={disabled}
        type="checkbox"
        onChange={(event) => onChange(event.target.checked)}
      />
      <span
        aria-hidden="true"
        className="relative h-[18px] w-8 rounded-full border border-app-border bg-app-surface transition-colors peer-checked:border-app-accent peer-checked:bg-app-accent after:absolute after:top-0.5 after:left-0.5 after:h-3 after:w-3 after:rounded-full after:bg-app-muted after:transition-[left,background-color] peer-checked:after:left-[17px] peer-checked:after:bg-app-surface"
      />
      {hideLabel ? null : <strong className="font-semibold">{label}</strong>}
    </label>
  );
  return <OptionalAppTooltip content={tooltip}>{control}</OptionalAppTooltip>;
}
