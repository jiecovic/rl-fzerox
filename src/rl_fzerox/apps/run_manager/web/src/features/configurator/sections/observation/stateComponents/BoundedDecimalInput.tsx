// src/rl_fzerox/apps/run_manager/web/src/features/configurator/sections/observation/stateComponents/BoundedDecimalInput.tsx
import { useEffect, useState } from "react";

import { formatDecimalInput } from "@/features/configurator/fields/format";

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
    if (raw.trim() === "") {
      return null;
    }
    const parsed = Number(raw);
    if (!Number.isFinite(parsed) || parsed < min || (max !== undefined && parsed > max)) {
      return null;
    }
    return Math.round(parsed * 100) / 100;
  }

  function tryCommitRaw(nextRawValue: string) {
    const parsed = parseBounded(nextRawValue);
    if (parsed === null) {
      return;
    }
    onChange(parsed);
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
      inputMode="decimal"
      max={max}
      min={min}
      step={step}
      type="number"
      value={rawValue}
      onBlur={commitValue}
      onChange={(event) => {
        const nextRawValue = event.target.value;
        setRawValue(nextRawValue);
        tryCommitRaw(nextRawValue);
      }}
      onKeyDown={(event) => {
        if (event.key === "Enter") {
          event.currentTarget.blur();
        }
      }}
    />
  );
}
