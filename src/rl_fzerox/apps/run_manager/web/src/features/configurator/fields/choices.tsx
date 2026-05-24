// src/rl_fzerox/apps/run_manager/web/src/features/configurator/fields/choices.tsx
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
  optionLabels = {},
  resetValue,
}: {
  help: string;
  label: string;
  value: T;
  options: readonly T[];
  onChange: (value: T) => void;
  optionLabels?: Partial<Record<T, string>>;
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
            {optionLabels[option] ?? option}
          </option>
        ))}
      </select>
    </div>
  );
}

export function SegmentedChoiceStrip({
  ariaLabel,
  options,
}: {
  ariaLabel: string;
  options: readonly {
    active: boolean;
    disabled?: boolean;
    key: string;
    label: string;
    tooltip?: string;
    onClick: () => void;
  }[];
}) {
  return (
    <fieldset className="segmented-choice-strip">
      <legend className="segmented-choice-strip-legend">{ariaLabel}</legend>
      {options.map((option) => (
        <button
          aria-label={option.label}
          aria-disabled={option.disabled === true}
          aria-pressed={option.active}
          className={
            option.active
              ? option.disabled
                ? "segmented-choice-chip active disabled tooltip-anchor"
                : "segmented-choice-chip active tooltip-anchor"
              : option.disabled
                ? "segmented-choice-chip disabled tooltip-anchor"
                : "segmented-choice-chip tooltip-anchor"
          }
          data-tooltip={option.tooltip}
          key={option.key}
          type="button"
          onClick={() => {
            if (!option.disabled) {
              option.onClick();
            }
          }}
        >
          {option.label}
        </button>
      ))}
    </fieldset>
  );
}
