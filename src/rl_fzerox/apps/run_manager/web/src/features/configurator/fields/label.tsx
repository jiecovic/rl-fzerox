// src/rl_fzerox/apps/run_manager/web/src/features/configurator/fields/label.tsx
import type { FieldLabelProps } from "@/features/configurator/fields/types";
import { IconButton } from "@/shared/ui/Button";
import { FieldLabelRow } from "@/shared/ui/Field";
import { HelpTooltipButton } from "@/shared/ui/HelpTooltipButton";
import { ResetIcon } from "@/shared/ui/icons";

export function FieldLabel({ help, label, onReset }: FieldLabelProps) {
  return (
    <FieldLabelRow>
      <span>{label}</span>
      <HelpTooltipButton label={label} text={help} />
      {onReset !== undefined ? (
        <IconButton
          aria-label={`Reset ${label} to default`}
          className="tooltip-anchor bg-transparent hover:bg-transparent"
          data-tooltip="Reset to default"
          size="micro"
          tone="muted"
          onClick={(event) => {
            event.preventDefault();
            onReset();
          }}
        >
          <ResetIcon />
        </IconButton>
      ) : null}
    </FieldLabelRow>
  );
}
