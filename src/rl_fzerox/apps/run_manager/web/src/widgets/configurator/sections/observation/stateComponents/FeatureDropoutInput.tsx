// src/rl_fzerox/apps/run_manager/web/src/widgets/configurator/sections/observation/stateComponents/FeatureDropoutInput.tsx
import { BoundedDecimalInput } from "@/widgets/configurator/sections/observation/stateComponents/BoundedDecimalInput";

interface FeatureDropoutInputProps {
  disabled: boolean;
  label: string;
  value: number;
  onChange: (value: number) => void;
}

export function FeatureDropoutInput({
  disabled,
  label,
  value,
  onChange,
}: FeatureDropoutInputProps) {
  return (
    <BoundedDecimalInput
      className="state-feature-dropout-input"
      disabled={disabled}
      label={label}
      max={1}
      min={0}
      step="0.01"
      value={value}
      onChange={onChange}
    />
  );
}
