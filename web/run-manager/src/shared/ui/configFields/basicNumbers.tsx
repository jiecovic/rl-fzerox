// web/run-manager/src/shared/ui/configFields/basicNumbers.tsx
import type { ReactNode } from "react";
import { formatEditableDecimal, formatInteger } from "@/shared/ui/configFields/format";
import { FieldLabel } from "@/shared/ui/configFields/label";
import {
  blurOnEnter,
  editableNumberInputProps,
  parseDecimalInput,
  parsePositiveScientificInput,
  parseSafeIntegerInput,
  useEditableNumberInput,
} from "@/shared/ui/configFields/numberInput";
import { resetHandler } from "@/shared/ui/configFields/reset";
import { FieldInput, FieldNote, FieldShell } from "@/shared/ui/Field";

interface IntegerTextInputProps {
  "aria-label": string;
  className?: string;
  disabled?: boolean;
  max?: number;
  min?: number;
  value: number;
  onChange: (value: number) => void;
}

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
  const input = useEditableNumberInput({
    format: formatEditableDecimal,
    formattedValue: formatEditableDecimal(value),
    normalize: (nextValue) => nextValue,
    onCommit: onChange,
    parse: parseDecimalInput,
  });

  return (
    <FieldShell>
      <FieldLabel help={help} label={label} onReset={resetHandler(value, resetValue, onChange)} />
      <FieldInput
        aria-label={label}
        className="min-w-[9ch] max-w-[14ch] justify-self-start"
        {...editableNumberInputProps("decimal")}
        value={input.rawValue}
        onBlur={input.commitRawValue}
        onChange={(event) => input.changeRawValue(event.target.value)}
        onKeyDown={blurOnEnter}
      />
    </FieldShell>
  );
}

export function IntegerField({
  help,
  label,
  note,
  min = 0,
  max,
  value,
  onChange,
  resetValue,
}: {
  help: string;
  label: string;
  note?: ReactNode;
  min?: number;
  max?: number;
  value: number;
  onChange: (value: number) => void;
  resetValue?: number;
}) {
  const input = useEditableNumberInput({
    format: formatInteger,
    formattedValue: formatInteger(value),
    normalize: Math.round,
    onCommit: onChange,
    parse: (rawValue) => parseSafeIntegerInput(rawValue, { max, min }),
  });

  return (
    <FieldShell>
      <FieldLabel help={help} label={label} onReset={resetHandler(value, resetValue, onChange)} />
      <FieldInput
        aria-label={label}
        className="min-w-[9ch] max-w-[14ch] justify-self-start"
        {...editableNumberInputProps("integer")}
        value={input.rawValue}
        onBlur={input.commitRawValue}
        onChange={(event) => input.changeRawValue(event.target.value)}
        onKeyDown={blurOnEnter}
      />
      {note !== undefined ? <FieldNote>{note}</FieldNote> : null}
    </FieldShell>
  );
}

export function IntegerTextInput({
  "aria-label": ariaLabel,
  className,
  disabled = false,
  max,
  min = 0,
  value,
  onChange,
}: IntegerTextInputProps) {
  const input = useEditableNumberInput({
    format: formatInteger,
    formattedValue: formatInteger(value),
    normalize: Math.round,
    onCommit: onChange,
    parse: (rawValue) => parseSafeIntegerInput(rawValue, { max, min }),
  });

  return (
    <input
      aria-label={ariaLabel}
      className={className}
      disabled={disabled}
      max={max}
      min={min}
      step={1}
      {...editableNumberInputProps("integer")}
      value={input.rawValue}
      onBlur={input.commitRawValue}
      onChange={(event) => input.changeRawValue(event.target.value)}
      onKeyDown={blurOnEnter}
    />
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
  const input = useEditableNumberInput({
    format: (nextValue) => nextValue.toExponential(2),
    formattedValue: value.toExponential(2),
    normalize: (nextValue) => nextValue,
    onCommit: onChange,
    parse: parsePositiveScientificInput,
  });

  return (
    <FieldShell>
      <FieldLabel help={help} label={label} onReset={resetHandler(value, resetValue, onChange)} />
      <FieldInput
        aria-label={label}
        className="min-w-[9ch] max-w-[14ch] justify-self-start"
        {...editableNumberInputProps("scientific")}
        value={input.rawValue}
        onBlur={input.commitRawValue}
        onChange={(event) => input.changeRawValue(event.target.value)}
        onKeyDown={blurOnEnter}
      />
    </FieldShell>
  );
}
