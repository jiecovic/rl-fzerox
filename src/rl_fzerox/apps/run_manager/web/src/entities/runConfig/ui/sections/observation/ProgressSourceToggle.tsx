// src/rl_fzerox/apps/run_manager/web/src/entities/runConfig/ui/sections/observation/ProgressSourceToggle.tsx

import {
  progressOptionLabel,
  progressSourceOptions,
} from "@/entities/runConfig/ui/sections/observation/featureRows";
import type { StateComponentConfig } from "@/shared/api/contract";
import { SegmentedChoiceButton, SegmentedChoiceGroup } from "@/shared/ui/Field";

interface ProgressSourceToggleProps {
  disabled: boolean;
  value: NonNullable<StateComponentConfig["progress_source"]>;
  onChange: (value: NonNullable<StateComponentConfig["progress_source"]>) => void;
}

export function ProgressSourceToggle({ disabled, value, onChange }: ProgressSourceToggleProps) {
  return (
    <SegmentedChoiceGroup
      className="rounded-md p-px"
      label="Progress scalar source"
      value={value}
      onValueChange={(nextValue) => {
        if (isProgressSource(nextValue)) {
          onChange(nextValue);
        }
      }}
    >
      {progressSourceOptions.map((option) => (
        <SegmentedChoiceButton
          active={option === value}
          className="min-h-7 px-2.5 text-xs"
          disabledChoice={disabled}
          key={option}
          value={option}
        >
          {progressOptionLabel(option)}
        </SegmentedChoiceButton>
      ))}
    </SegmentedChoiceGroup>
  );
}

function isProgressSource(
  value: string,
): value is NonNullable<StateComponentConfig["progress_source"]> {
  return progressSourceOptions.some((option) => option === value);
}
