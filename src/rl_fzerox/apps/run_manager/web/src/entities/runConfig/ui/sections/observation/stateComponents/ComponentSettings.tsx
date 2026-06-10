// src/rl_fzerox/apps/run_manager/web/src/entities/runConfig/ui/sections/observation/stateComponents/ComponentSettings.tsx

import type { StateComponentConfig } from "@/shared/api/contract";
import { IntegerField } from "@/shared/ui/configFields";

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
    <div className="grid grid-cols-[repeat(auto-fit,minmax(190px,240px))] gap-3">
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
      {disabled ? (
        <span className="self-end text-[13px] text-app-muted">category disabled</span>
      ) : null}
    </div>
  );
}
