// src/rl_fzerox/apps/run_manager/web/src/features/configurator/sections/observation/ProgressSourceToggle.tsx
import type { StateComponentConfig } from "@/shared/api/contract";

import { progressOptionLabel, progressSourceOptions } from "./featureRows";

interface ProgressSourceToggleProps {
  disabled: boolean;
  value: NonNullable<StateComponentConfig["progress_source"]>;
  onChange: (value: NonNullable<StateComponentConfig["progress_source"]>) => void;
}

export function ProgressSourceToggle({ disabled, value, onChange }: ProgressSourceToggleProps) {
  return (
    <fieldset className="state-progress-toggle" disabled={disabled}>
      <legend>Progress scalar source</legend>
      {progressSourceOptions.map((option) => (
        <button
          className={option === value ? "active" : undefined}
          key={option}
          type="button"
          onClick={() => onChange(option)}
        >
          {progressOptionLabel(option)}
        </button>
      ))}
    </fieldset>
  );
}
