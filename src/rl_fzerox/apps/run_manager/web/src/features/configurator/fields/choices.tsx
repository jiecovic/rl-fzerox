import { FieldLabel } from "@/features/configurator/fields/label";
import { resetHandler } from "@/features/configurator/fields/reset";

export function BooleanField({
  help,
  label,
  value,
  onChange,
  resetValue,
}: {
  help: string;
  label: string;
  value: boolean;
  onChange: (value: boolean) => void;
  resetValue?: boolean;
}) {
  return (
    <div className="field-shell boolean-field">
      <FieldLabel help={help} label={label} onReset={resetHandler(value, resetValue, onChange)} />
      <button
        aria-label={label}
        aria-pressed={value}
        className={value ? "switch-button active" : "switch-button"}
        type="button"
        onClick={() => onChange(!value)}
      >
        <span aria-hidden="true" />
        <strong>{value ? "On" : "Off"}</strong>
      </button>
    </div>
  );
}

export function SelectField<T extends string>({
  help,
  label,
  value,
  options,
  onChange,
  resetValue,
}: {
  help: string;
  label: string;
  value: T;
  options: readonly T[];
  onChange: (value: T) => void;
  resetValue?: T;
}) {
  return (
    <div className="field-shell">
      <FieldLabel help={help} label={label} onReset={resetHandler(value, resetValue, onChange)} />
      <select
        aria-label={label}
        value={value}
        onChange={(event) => onChange(event.target.value as T)}
      >
        {options.map((option) => (
          <option key={option} value={option}>
            {option}
          </option>
        ))}
      </select>
    </div>
  );
}
