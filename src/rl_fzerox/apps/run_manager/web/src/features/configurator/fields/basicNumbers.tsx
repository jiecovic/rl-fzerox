import { useEffect, useState } from "react";

import { formatInteger } from "@/features/configurator/fields/format";
import { FieldLabel } from "@/features/configurator/fields/label";
import { resetHandler } from "@/features/configurator/fields/reset";

export function NumberField({
  help,
  label,
  value,
  onChange,
  resetValue,
  step = "1",
}: {
  help: string;
  label: string;
  value: number;
  onChange: (value: number) => void;
  resetValue?: number;
  step?: string;
}) {
  return (
    <div className="field-shell compact-value-field">
      <FieldLabel help={help} label={label} onReset={resetHandler(value, resetValue, onChange)} />
      <input
        aria-label={label}
        type="number"
        step={step}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
      />
    </div>
  );
}

export function IntegerField({
  help,
  label,
  min = 0,
  value,
  onChange,
  resetValue,
}: {
  help: string;
  label: string;
  min?: number;
  value: number;
  onChange: (value: number) => void;
  resetValue?: number;
}) {
  const [rawValue, setRawValue] = useState(formatInteger(value));

  useEffect(() => {
    setRawValue(formatInteger(value));
  }, [value]);

  function commitValue() {
    const parsed = Number(rawValue.replace(/[,_\s]/g, ""));
    if (!Number.isSafeInteger(parsed) || parsed < min) {
      setRawValue(formatInteger(value));
      return;
    }
    onChange(parsed);
    setRawValue(formatInteger(parsed));
  }

  return (
    <div className="field-shell compact-value-field">
      <FieldLabel help={help} label={label} onReset={resetHandler(value, resetValue, onChange)} />
      <input
        aria-label={label}
        inputMode="numeric"
        spellCheck={false}
        value={rawValue}
        onBlur={commitValue}
        onChange={(event) => setRawValue(event.target.value)}
      />
    </div>
  );
}

export function ScientificNumberField({
  help,
  label,
  value,
  onChange,
  resetValue,
}: {
  help: string;
  label: string;
  value: number;
  onChange: (value: number) => void;
  resetValue?: number;
}) {
  const [rawValue, setRawValue] = useState(value.toExponential(2));

  useEffect(() => {
    setRawValue(value.toExponential(2));
  }, [value]);

  function commitValue() {
    const parsed = Number(rawValue);
    if (!Number.isFinite(parsed) || parsed <= 0) {
      setRawValue(value.toExponential(2));
      return;
    }
    onChange(parsed);
    setRawValue(parsed.toExponential(2));
  }

  return (
    <div className="field-shell compact-value-field">
      <FieldLabel help={help} label={label} onReset={resetHandler(value, resetValue, onChange)} />
      <input
        aria-label={label}
        inputMode="decimal"
        spellCheck={false}
        value={rawValue}
        onBlur={commitValue}
        onChange={(event) => setRawValue(event.target.value)}
      />
    </div>
  );
}
