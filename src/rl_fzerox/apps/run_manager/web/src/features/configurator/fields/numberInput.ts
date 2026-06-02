// src/rl_fzerox/apps/run_manager/web/src/features/configurator/fields/numberInput.ts
import { useEffect, useState } from "react";

interface EditableNumberInputOptions {
  format: (value: number) => string;
  formattedValue: string;
  normalize: (value: number) => number;
  onCommit: (value: number) => void;
  onValidInput?: (value: number) => void;
  parse: (rawValue: string) => number | null;
}

export function useEditableNumberInput({
  format,
  formattedValue,
  normalize,
  onCommit,
  onValidInput,
  parse,
}: EditableNumberInputOptions) {
  const [rawValue, setRawValue] = useState(formattedValue);

  useEffect(() => {
    setRawValue(formattedValue);
  }, [formattedValue]);

  function changeRawValue(nextRawValue: string) {
    setRawValue(nextRawValue);
    const parsed = parse(nextRawValue);
    if (parsed === null) {
      return;
    }
    onValidInput?.(normalize(parsed));
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
