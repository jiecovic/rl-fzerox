// src/rl_fzerox/apps/run_manager/web/src/widgets/configurator/fields/numberInput.ts
import type { InputHTMLAttributes, KeyboardEvent } from "react";
import { useEffect, useState } from "react";

export type EditableNumberKind = "decimal" | "integer" | "scientific";

interface NumberInputBounds {
  max?: number;
  min?: number;
}

interface EditableNumberInputOptions {
  format: (value: number) => string;
  formattedValue: string;
  normalize: (value: number) => number;
  onCommit: (value: number) => void;
  parse: (rawValue: string) => number | null;
}

export function useEditableNumberInput({
  format,
  formattedValue,
  normalize,
  onCommit,
  parse,
}: EditableNumberInputOptions) {
  const [rawValue, setRawValue] = useState(formattedValue);

  useEffect(() => {
    setRawValue(formattedValue);
  }, [formattedValue]);

  function changeRawValue(nextRawValue: string) {
    setRawValue(nextRawValue);
  }

  function commitRawValue() {
    const parsed = parse(rawValue);
    if (parsed === null) {
      setRawValue(formattedValue);
      return;
    }
    setCommittedValue(parsed);
  }

  function setCommittedValue(nextValue: number) {
    const normalized = normalize(nextValue);
    onCommit(normalized);
    setRawValue(format(normalized));
  }

  return {
    changeRawValue,
    commitRawValue,
    rawValue,
    setCommittedValue,
    setRawValue,
  };
}

export function editableNumberInputProps(
  kind: EditableNumberKind,
): Pick<InputHTMLAttributes<HTMLInputElement>, "inputMode" | "spellCheck"> {
  return {
    inputMode: kind === "integer" ? "numeric" : "decimal",
    spellCheck: false,
  };
}

export function blurOnEnter(event: KeyboardEvent<HTMLInputElement>) {
  if (event.key === "Enter") {
    event.currentTarget.blur();
  }
}

export function parseDecimalInput(rawValue: string, bounds: NumberInputBounds = {}) {
  const trimmed = rawValue.trim();
  if (isEmptyOrPartialNumber(trimmed)) {
    return null;
  }
  const parsed = Number(trimmed);
  return Number.isFinite(parsed) && withinBounds(parsed, bounds) ? parsed : null;
}

export function parsePositiveScientificInput(rawValue: string, bounds: NumberInputBounds = {}) {
  const parsed = parseDecimalInput(rawValue, { min: Math.max(bounds.min ?? 0, Number.EPSILON) });
  if (parsed === null || !withinBounds(parsed, bounds)) {
    return null;
  }
  return parsed;
}

export function parseSafeIntegerInput(rawValue: string, bounds: NumberInputBounds = {}) {
  const normalized = rawValue.replace(/[,_\s]/gu, "");
  if (normalized === "" || normalized === "-" || normalized === "+") {
    return null;
  }
  const parsed = Number(normalized);
  return Number.isSafeInteger(parsed) && withinBounds(parsed, bounds) ? parsed : null;
}

function isEmptyOrPartialNumber(value: string) {
  return (
    value === "" ||
    value === "-" ||
    value === "+" ||
    value === "." ||
    value === "-." ||
    value === "+."
  );
}

function withinBounds(value: number, { max, min }: NumberInputBounds) {
  return (min === undefined || value >= min) && (max === undefined || value <= max);
}
