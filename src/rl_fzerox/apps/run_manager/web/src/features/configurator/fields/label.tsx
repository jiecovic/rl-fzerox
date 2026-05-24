// src/rl_fzerox/apps/run_manager/web/src/features/configurator/fields/label.tsx
import type { FieldLabelProps } from "@/features/configurator/fields/types";
import { HelpTooltipButton } from "@/shared/ui/HelpTooltipButton";
import { ResetIcon } from "@/shared/ui/icons";

export function FieldLabel({ help, label, onReset }: FieldLabelProps) {
  return (
    <span className="field-label">
      <span>{label}</span>
      <HelpTooltipButton label={label} text={help} />
      {onReset !== undefined ? (
        <button
          aria-label={`Reset ${label} to default`}
          className="field-reset-button tooltip-anchor"
          data-tooltip="Reset to default"
          type="button"
          onClick={(event) => {
            event.preventDefault();
            onReset();
          }}
        >
          <ResetIcon />
        </button>
      ) : null}
    </span>
  );
}
