// src/rl_fzerox/apps/run_manager/web/src/features/configurator/sections/observation/stateComponents/ComponentSettings.tsx
import { IntegerField } from "@/features/configurator/fields";
import type { StateComponentConfig } from "@/shared/api/contract";

interface ComponentSettingsProps {
  checkpointLocked?: boolean;
  component: StateComponentConfig;
  disabled: boolean;
  updateComponent: (name: string, patch: Partial<StateComponentConfig>) => void;
}

export function ComponentSettings({
  checkpointLocked = false,
  component,
  disabled,
  updateComponent,
}: ComponentSettingsProps) {
  const showSettings = component.name === "control_history" || disabled;
  if (!showSettings) {
    return null;
  }
  return (
    <div className="state-component-settings">
      {component.name === "control_history" ? (
        <fieldset className="fork-lock-fieldset" disabled={checkpointLocked}>
          <IntegerField
            help="Number of prior action samples exposed in the state vector."
            label="History length"
            min={1}
            value={component.length ?? 1}
            onChange={(value) => updateComponent(component.name, { length: value })}
          />
        </fieldset>
      ) : null}
      {disabled ? <span className="state-component-disabled">category disabled</span> : null}
    </div>
  );
}
