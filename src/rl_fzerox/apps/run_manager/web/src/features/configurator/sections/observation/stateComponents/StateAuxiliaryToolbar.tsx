// src/rl_fzerox/apps/run_manager/web/src/features/configurator/sections/observation/stateComponents/StateAuxiliaryToolbar.tsx
import { ToggleSwitch } from "@/features/configurator/fields";

interface StateAuxiliaryToolbarProps {
  enabled: boolean;
  onChange: (enabled: boolean) => void;
}

export function StateAuxiliaryToolbar({ enabled, onChange }: StateAuxiliaryToolbarProps) {
  return (
    <div className="state-auxiliary-toolbar">
      <span className="state-auxiliary-toolbar-copy">
        <strong>Auxiliary RAM supervision</strong>
        <small>Optional supervised targets over the shared policy latent.</small>
      </span>
      <ToggleSwitch
        checked={enabled}
        disabled={false}
        hideLabel
        label="auxiliary state supervision enabled"
        tooltip={
          enabled
            ? "Disable auxiliary RAM supervision and clear active aux losses"
            : "Enable auxiliary RAM supervision"
        }
        onChange={onChange}
      />
    </div>
  );
}
