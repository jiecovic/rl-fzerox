// src/rl_fzerox/apps/run_manager/web/src/widgets/configurator/sections/action/ActionToggleRow.tsx
import { ToggleSwitch } from "@/widgets/configurator/fields";
import {
  ActionToggleCopy,
  ActionToggleRowLayout,
} from "@/widgets/configurator/sections/action/ActionLayout";

export function ActionToggleRow({
  description,
  enabled,
  enabledDisabledReason,
  enabledLabel,
  label,
  onEnabledChange,
  onOutputChange,
  output,
  outputDisabledReason,
  outputLabel,
}: {
  description: string;
  enabled: boolean;
  enabledDisabledReason?: string;
  enabledLabel: string;
  label: string;
  onEnabledChange: (checked: boolean) => void;
  onOutputChange: (checked: boolean) => void;
  output: boolean;
  outputDisabledReason?: string;
  outputLabel: string;
}) {
  return (
    <ActionToggleRowLayout>
      <ActionToggleCopy description={description} title={label} />
      <ToggleSwitch
        checked={output}
        disabled={outputDisabledReason !== undefined}
        hideLabel
        label={outputLabel}
        tooltip={outputDisabledReason ?? `Keep ${label.toLowerCase()} in the action output`}
        onChange={onOutputChange}
      />
      <ToggleSwitch
        checked={enabled}
        disabled={!output || enabledDisabledReason !== undefined}
        hideLabel
        label={enabledLabel}
        tooltip={
          enabledDisabledReason ??
          (!output
            ? `Add ${label.toLowerCase()} to the action output first`
            : `Mask or unmask ${label.toLowerCase()} at runtime`)
        }
        onChange={onEnabledChange}
      />
    </ActionToggleRowLayout>
  );
}
