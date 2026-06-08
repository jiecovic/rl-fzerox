// src/rl_fzerox/apps/run_manager/web/src/features/configurator/sections/tracks/ChoiceStrip.tsx
import { SegmentedChoiceStrip } from "@/features/configurator/fields";
import { cn } from "@/shared/ui/cn";
import { OptionalAppTooltip } from "@/shared/ui/Tooltip";

interface ChoiceOption {
  active: boolean;
  disabled?: boolean;
  key: string;
  label: string;
  tooltip?: string;
  onClick: () => void;
}

export function ChoiceStrip({
  description,
  options,
}: {
  description: string;
  options: readonly ChoiceOption[];
}) {
  return (
    <div className="grid gap-2.5">
      <SegmentedChoiceStrip ariaLabel="Selection" options={options} />
      <p className="m-0 text-xs leading-snug text-app-muted">{description}</p>
    </div>
  );
}

export function ToggleChoiceStrip({
  description,
  options,
}: {
  description: string;
  options: readonly ChoiceOption[];
}) {
  return (
    <div className="grid gap-2.5">
      <fieldset className="m-0 inline-flex w-fit max-w-full min-w-0 flex-wrap justify-self-start overflow-hidden rounded-lg border border-app-border bg-app-surface-muted p-0.5">
        <legend className="sr-only">Selection</legend>
        {options.map((option) => (
          <OptionalAppTooltip content={option.tooltip} key={option.key}>
            <button
              aria-pressed={option.active}
              className={cn(
                "min-h-8 min-w-0 rounded-md border border-transparent px-3.5 text-sm font-medium whitespace-nowrap text-app-muted transition-colors",
                option.active ? "border-app-border-strong bg-app-surface text-app-text" : undefined,
                option.disabled ? "cursor-not-allowed opacity-55" : undefined,
                !option.active && option.disabled !== true
                  ? "hover:bg-[color-mix(in_srgb,var(--accent)_8%,var(--surface-muted))] hover:text-app-text"
                  : undefined,
              )}
              disabled={option.disabled}
              type="button"
              onClick={option.onClick}
            >
              {option.label}
            </button>
          </OptionalAppTooltip>
        ))}
      </fieldset>
      <p className="m-0 text-xs leading-snug text-app-muted">{description}</p>
    </div>
  );
}
