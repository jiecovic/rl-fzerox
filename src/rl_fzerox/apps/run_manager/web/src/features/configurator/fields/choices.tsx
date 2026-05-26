// src/rl_fzerox/apps/run_manager/web/src/features/configurator/fields/choices.tsx
import { FieldLabel } from "@/features/configurator/fields/label";
import { resetHandler } from "@/features/configurator/fields/reset";
import {
  FieldSelect,
  FieldShell,
  SegmentedChoiceButton,
  SegmentedChoiceGroup,
  SwitchButton,
} from "@/shared/ui/Field";

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
    <FieldShell>
      <FieldLabel help={help} label={label} onReset={resetHandler(value, resetValue, onChange)} />
      <SwitchButton active={value} label={label} onClick={() => onChange(!value)} />
    </FieldShell>
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
    <FieldShell>
      <FieldLabel help={help} label={label} onReset={resetHandler(value, resetValue, onChange)} />
      <FieldSelect
        aria-label={label}
        value={value}
        onChange={(event) => onChange(event.target.value as T)}
      >
        {options.map((option) => (
          <option key={option} value={option}>
            {optionLabels[option] ?? option}
          </option>
        ))}
      </FieldSelect>
    </FieldShell>
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
    <SegmentedChoiceGroup>
      <legend className="sr-only">{ariaLabel}</legend>
      {options.map((option) => (
        <SegmentedChoiceButton
          active={option.active}
          aria-label={option.label}
          aria-disabled={option.disabled === true}
          aria-pressed={option.active}
          className="tooltip-anchor"
          data-tooltip={option.tooltip}
          disabledChoice={option.disabled}
          key={option.key}
          onClick={() => {
            if (!option.disabled) {
              option.onClick();
            }
          }}
        >
          {option.label}
        </SegmentedChoiceButton>
      ))}
    </SegmentedChoiceGroup>
  );
}
