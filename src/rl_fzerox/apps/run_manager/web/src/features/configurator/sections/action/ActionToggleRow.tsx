// src/rl_fzerox/apps/run_manager/web/src/features/configurator/sections/action/ActionToggleRow.tsx
import { ToggleSwitch } from "@/features/configurator/fields";

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
    <div className="action-toggle-row">
      <div className="action-toggle-copy">
        <strong>{label}</strong>
        <small>{description}</small>
      </div>
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
    </div>
  );
}
