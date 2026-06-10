// src/rl_fzerox/apps/run_manager/web/src/widgets/configurator/fields/label.tsx

import { FieldLabelRow } from "@/shared/ui/Field";
import { HelpTooltipButton } from "@/shared/ui/HelpTooltipButton";
import { ResetIcon } from "@/shared/ui/icons";
import { TooltipIconButton } from "@/shared/ui/TooltipIconButton";
import type { FieldLabelProps } from "@/widgets/configurator/fields/types";

export function FieldLabel({ help, label, onReset }: FieldLabelProps) {
  return (
    <FieldLabelRow>
      <span>{label}</span>
      <HelpTooltipButton label={label} text={help} />
      {onReset !== undefined ? (
        <TooltipIconButton
          aria-label={`Reset ${label} to default`}
          className="bg-transparent hover:bg-transparent"
          size="micro"
          tone="muted"
          tooltip="Reset to default"
          onClick={(event) => {
            event.preventDefault();
            onReset();
          }}
        >
          <ResetIcon />
        </TooltipIconButton>
      ) : null}
    </FieldLabelRow>
  );
}
