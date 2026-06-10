// web/run-manager/src/shared/ui/configFields/choices.tsx

import { FieldLabel } from "@/shared/ui/configFields/label";
import { resetHandler } from "@/shared/ui/configFields/reset";
import {
  FieldSelect,
  FieldShell,
  SegmentedChoiceButton,
  SegmentedChoiceGroup,
  SwitchButton,
} from "@/shared/ui/Field";
import { OptionalAppTooltip } from "@/shared/ui/Tooltip";

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
  const activeOption = options.find((option) => option.active);

  return (
    <SegmentedChoiceGroup
      label={ariaLabel}
      value={activeOption?.key ?? ""}
      onValueChange={(nextValue) => {
        const nextOption = options.find((option) => option.key === nextValue);
        if (nextOption !== undefined && nextOption.disabled !== true) {
          nextOption.onClick();
        }
      }}
    >
      {options.map((option) => (
        <OptionalAppTooltip content={option.tooltip} key={option.key}>
          <SegmentedChoiceButton
            active={option.active}
            disabledChoice={option.disabled}
            value={option.key}
          >
            {option.label}
          </SegmentedChoiceButton>
        </OptionalAppTooltip>
      ))}
    </SegmentedChoiceGroup>
  );
}
