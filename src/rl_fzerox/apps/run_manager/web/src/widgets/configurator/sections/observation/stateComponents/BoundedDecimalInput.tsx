// src/rl_fzerox/apps/run_manager/web/src/widgets/configurator/sections/observation/stateComponents/BoundedDecimalInput.tsx
import { useEffect, useState } from "react";

import { formatDecimalInput } from "@/widgets/configurator/fields/format";
import {
  blurOnEnter,
  editableNumberInputProps,
  parseDecimalInput,
} from "@/widgets/configurator/fields/numberInput";

interface BoundedDecimalInputProps {
  className: string;
  disabled: boolean;
  label: string;
  max?: number;
  min: number;
  step: string;
  value: number;
  onChange: (value: number) => void;
}

export function BoundedDecimalInput({
  className,
  disabled,
  label,
  max,
  min,
  step,
  value,
  onChange,
}: BoundedDecimalInputProps) {
  const [rawValue, setRawValue] = useState(formatDecimalInput(value, step));

  useEffect(() => {
    setRawValue(formatDecimalInput(value, step));
  }, [step, value]);

  function parseBounded(raw: string) {
    const parsed = parseDecimalInput(raw, { max, min });
    if (parsed === null) {
      return null;
    }
    return Math.round(parsed * 100) / 100;
  }

  function commitValue() {
    const parsed = parseBounded(rawValue);
    if (parsed === null) {
      setRawValue(formatDecimalInput(value, step));
      return;
    }
    onChange(parsed);
    setRawValue(formatDecimalInput(parsed, step));
  }

  return (
    <input
      aria-label={label}
      className={className}
      disabled={disabled}
      max={max}
      min={min}
      step={step}
      {...editableNumberInputProps("decimal")}
      value={rawValue}
      onBlur={commitValue}
      onChange={(event) => setRawValue(event.target.value)}
      onKeyDown={blurOnEnter}
    />
  );
}
