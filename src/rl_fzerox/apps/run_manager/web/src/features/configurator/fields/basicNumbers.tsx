// src/rl_fzerox/apps/run_manager/web/src/features/configurator/fields/basicNumbers.tsx
import { useEffect, useState } from "react";

import { formatEditableDecimal, formatInteger } from "@/features/configurator/fields/format";
import { FieldLabel } from "@/features/configurator/fields/label";
import { resetHandler } from "@/features/configurator/fields/reset";

export function NumberField({
  help,
  label,
  value,
  onChange,
  resetValue,
  step: _step = "1",
}: {
  help: string;
  label: string;
  value: number;
  onChange: (value: number) => void;
  resetValue?: number;
  step?: string;
}) {
  const [rawValue, setRawValue] = useState(formatEditableDecimal(value));

  useEffect(() => {
    setRawValue(formatEditableDecimal(value));
  }, [value]);

  function commitValue() {
    const normalized = rawValue.trim();
    if (normalized === "" || normalized === "-" || normalized === "." || normalized === "-.") {
      setRawValue(formatEditableDecimal(value));
      return;
    }
    const parsed = Number(normalized);
    if (!Number.isFinite(parsed)) {
      setRawValue(formatEditableDecimal(value));
      return;
    }
    onChange(parsed);
    setRawValue(formatEditableDecimal(parsed));
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

export function IntegerField({
  help,
  label,
  min = 0,
  max,
  value,
  onChange,
  resetValue,
}: {
  help: string;
  label: string;
  min?: number;
  max?: number;
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
    if (!Number.isSafeInteger(parsed) || parsed < min || (max !== undefined && parsed > max)) {
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
