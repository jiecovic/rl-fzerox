// src/rl_fzerox/apps/run_manager/web/src/shared/ui/configFields/rangeNumbers.tsx
import { useEffect, useState } from "react";
import {
  clamp,
  formatCompactDecimal,
  formatDecimalInput,
  formatInteger,
  roundSignificant,
  roundToStepPrecision,
} from "@/shared/ui/configFields/format";
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
import {
  discreteSliderTicks,
  nearestIndex,
  nearestOption,
  Slider,
} from "@/shared/ui/configFields/slider";
import type { SliderTick } from "@/shared/ui/configFields/types";
import {
  FieldInput,
  FieldShell,
  optionalNumberRowClass,
  RangeRow,
  SwitchButton,
} from "@/shared/ui/Field";

export function RangeNumberField({
  help,
  label,
  value,
  onChange,
  min,
  max,
  rangeStep,
  numberStep = String(rangeStep),
  resetValue,
  ticks = [],
}: {
  help: string;
  label: string;
  value: number;
  onChange: (value: number) => void;
  min: number;
  max: number;
  rangeStep: number;
  numberStep?: string;
  resetValue?: number;
  ticks?: readonly SliderTick[];
}) {
  const manualInput = useEditableNumberInput({
    format: (nextValue) => formatDecimalInput(nextValue, numberStep),
    formattedValue: formatDecimalInput(value, numberStep),
    normalize: (nextValue) => roundToStepPrecision(nextValue, numberStep),
    onCommit: onChange,
    parse: (rawValue) => parseDecimalInput(rawValue, { max, min }),
  });

  function updateFromSlider(nextValue: number) {
    const normalized = roundToStepPrecision(nextValue, rangeStep);
    onChange(normalized);
    manualInput.setRawValue(formatDecimalInput(normalized, numberStep));
  }

  return (
    <FieldShell>
      <FieldLabel help={help} label={label} onReset={resetHandler(value, resetValue, onChange)} />
      <RangeRow>
        <Slider
          ariaLabel={`${label} slider`}
          max={max}
          min={min}
          step={rangeStep}
          ticks={ticks}
          value={clamp(value, min, max)}
          valueLabel={formatDecimalInput(value, numberStep)}
          onChange={updateFromSlider}
        />
        <FieldInput
          aria-label={label}
          className="h-[34px] indent-0 tabular-nums"
          max={max}
          min={min}
          step={numberStep}
          {...editableNumberInputProps("decimal")}
          value={manualInput.rawValue}
          onBlur={manualInput.commitRawValue}
          onChange={(event) => manualInput.changeRawValue(event.target.value)}
          onKeyDown={blurOnEnter}
        />
      </RangeRow>
    </FieldShell>
  );
}

export function RangeIntegerField({
  help,
  label,
  value,
  onChange,
  min,
  max,
  rangeStep,
  resetValue,
  ticks = [],
}: {
  help: string;
  label: string;
  value: number;
  onChange: (value: number) => void;
  min: number;
  max: number;
  rangeStep: number;
  resetValue?: number;
  ticks?: readonly SliderTick[];
}) {
  const manualInput = useEditableNumberInput({
    format: formatInteger,
    formattedValue: formatInteger(value),
    normalize: Math.round,
    onCommit: onChange,
    parse: (rawValue) => parseSafeIntegerInput(rawValue, { max, min }),
  });

  function updateFromSlider(nextValue: number) {
    manualInput.setCommittedValue(nextValue);
  }

  return (
    <FieldShell>
      <FieldLabel
        help={help}
        label={label}
        onReset={resetHandler(value, resetValue, updateFromSlider)}
      />
      <RangeRow>
        <Slider
          ariaLabel={`${label} slider`}
          max={max}
          min={min}
          step={rangeStep}
          ticks={ticks}
          value={clamp(value, min, max)}
          valueLabel={formatInteger(value)}
          onChange={updateFromSlider}
        />
        <FieldInput
          aria-label={label}
          className="h-[34px] indent-0 tabular-nums"
          {...editableNumberInputProps("integer")}
          value={manualInput.rawValue}
          onBlur={manualInput.commitRawValue}
          onChange={(event) => manualInput.changeRawValue(event.target.value)}
          onKeyDown={blurOnEnter}
        />
      </RangeRow>
    </FieldShell>
  );
}

export function DiscreteSliderNumberField({
  help,
  label,
  value,
  onChange,
  sliderValues,
  maxManual,
  minManual = 1,
  resetValue,
  snapManualToOptions = false,
}: {
  help: string;
  label: string;
  value: number;
  onChange: (value: number) => void;
  sliderValues: readonly number[];
  maxManual: number;
  minManual?: number;
  resetValue?: number;
  snapManualToOptions?: boolean;
}) {
  const [rawValue, setRawValue] = useState(String(value));
  const sliderIndex = nearestIndex(value, sliderValues);
  const ticks = discreteSliderTicks(sliderValues);

  useEffect(() => {
    setRawValue(String(value));
  }, [value]);

  function commitManualValue() {
    const parsed = parseSafeIntegerInput(rawValue, { max: maxManual, min: minManual });
    if (parsed === null) {
      setRawValue(String(value));
      return;
    }
    const committed = snapManualToOptions ? nearestOption(parsed, sliderValues) : parsed;
    onChange(committed);
    setRawValue(String(committed));
  }

  function changeManualValue(nextRawValue: string) {
    setRawValue(nextRawValue);
  }

  function updateFromSlider(index: number) {
    const nextValue = sliderValues[index] ?? value;
    onChange(nextValue);
    setRawValue(String(nextValue));
  }

  return (
    <FieldShell>
      <FieldLabel help={help} label={label} onReset={resetHandler(value, resetValue, onChange)} />
      <RangeRow>
        <Slider
          ariaLabel={`${label} slider`}
          max={sliderValues.length - 1}
          min={0}
          step={1}
          ticks={ticks}
          value={sliderIndex}
          valueLabel={String(value)}
          onChange={updateFromSlider}
        />
        <FieldInput
          aria-label={label}
          className="h-[34px] indent-0 tabular-nums"
          max={maxManual}
          min={minManual}
          {...editableNumberInputProps("integer")}
          step="1"
          value={rawValue}
          onBlur={commitManualValue}
          onChange={(event) => changeManualValue(event.target.value)}
          onKeyDown={blurOnEnter}
        />
      </RangeRow>
    </FieldShell>
  );
}

export function LogRangeNumberField({
  help,
  label,
  value,
  onChange,
  min,
  max,
  resetValue,
  ticks = [],
}: {
  help: string;
  label: string;
  value: number;
  onChange: (value: number) => void;
  min: number;
  max: number;
  resetValue?: number;
  ticks?: readonly SliderTick[];
}) {
  const logMin = Math.log10(min);
  const logMax = Math.log10(max);
  const logValue = clamp(Math.log10(value), logMin, logMax);
  const input = useEditableNumberInput({
    format: (nextValue) => nextValue.toExponential(2),
    formattedValue: value.toExponential(2),
    normalize: (nextValue) => roundSignificant(nextValue, 3),
    onCommit: onChange,
    parse: (rawValue) => parsePositiveScientificInput(rawValue, { max, min }),
  });

  function updateFromSlider(nextLogValue: number) {
    onChange(roundSignificant(10 ** nextLogValue, 3));
  }

  return (
    <FieldShell>
      <FieldLabel help={help} label={label} onReset={resetHandler(value, resetValue, onChange)} />
      <RangeRow>
        <Slider
          ariaLabel={`${label} slider`}
          max={logMax}
          min={logMin}
          step={0.01}
          ticks={ticks.map((tick) => ({ label: tick.label, value: Math.log10(tick.value) }))}
          value={logValue}
          valueLabel={input.rawValue}
          onChange={updateFromSlider}
        />
        <FieldInput
          aria-label={label}
          className="h-[34px] indent-0 tabular-nums"
          {...editableNumberInputProps("scientific")}
          value={input.rawValue}
          onBlur={input.commitRawValue}
          onChange={(event) => input.changeRawValue(event.target.value)}
          onKeyDown={blurOnEnter}
        />
      </RangeRow>
    </FieldShell>
  );
}

export function OptionalNumberField({
  help,
  label,
  value,
  onChange,
  defaultValue = 0.1,
  max = 1,
  min = 0,
  resetValue,
  step = "0.01",
  enabledLabel = "On",
  nullLabel = "Off",
  sliderNullPosition = "none",
  sliderNullTickLabel = "∞",
}: {
  help: string;
  label: string;
  value: number | null;
  onChange: (value: number | null) => void;
  defaultValue?: number;
  max?: number;
  min?: number;
  resetValue?: number | null;
  step?: string;
  enabledLabel?: string;
  nullLabel?: string;
  sliderNullPosition?: "none" | "max";
  sliderNullTickLabel?: string;
}) {
  const enabled = value !== null;
  const [rawValue, setRawValue] = useState(value === null ? "" : formatDecimalInput(value, step));
  const sliderStep = Number(step);
  const sliderSupportsNullAtMax =
    sliderNullPosition === "max" && Number.isFinite(sliderStep) && sliderStep > 0;
  const sliderMax = sliderSupportsNullAtMax ? max + sliderStep : max;
  const sliderValue = enabled ? value : sliderMax;
  const sliderDisabled = !enabled && !sliderSupportsNullAtMax;
  const sliderTicks = sliderSupportsNullAtMax
    ? [
        { value: min, label: formatCompactDecimal(min) },
        { value: sliderMax, label: sliderNullTickLabel },
      ]
    : [
        { value: min, label: formatCompactDecimal(min) },
        { value: max, label: formatCompactDecimal(max) },
      ];

  useEffect(() => {
    setRawValue(value === null ? "" : formatDecimalInput(value, step));
  }, [step, value]);

  function updateFromSlider(nextValue: number) {
    if (sliderSupportsNullAtMax && nextValue >= sliderMax) {
      onChange(null);
      return;
    }
    const normalized = roundToStepPrecision(nextValue, step);
    onChange(normalized);
    setRawValue(formatDecimalInput(normalized, step));
  }

  function commitValue() {
    if (!enabled) {
      return;
    }
    const parsed = parseDecimalInput(rawValue, { max, min });
    if (parsed === null) {
      setRawValue(formatDecimalInput(value, step));
      return;
    }
    const normalized = roundToStepPrecision(parsed, step);
    onChange(normalized);
    setRawValue(formatDecimalInput(normalized, step));
  }

  function changeValue(nextRawValue: string) {
    setRawValue(nextRawValue);
  }

  return (
    <FieldShell>
      <FieldLabel help={help} label={label} onReset={resetHandler(value, resetValue, onChange)} />
      <div className={optionalNumberRowClass(sliderDisabled)}>
        <SwitchButton
          active={enabled}
          activeLabel={enabledLabel}
          inactiveLabel={nullLabel}
          label={`${label} enabled`}
          onClick={() => {
            if (enabled) {
              onChange(null);
              return;
            }
            const normalized = roundToStepPrecision(defaultValue, step);
            onChange(normalized);
            setRawValue(formatDecimalInput(normalized, step));
          }}
        />
        <Slider
          ariaLabel={`${label} slider`}
          disabled={sliderDisabled}
          max={sliderMax}
          min={min}
          step={sliderStep}
          ticks={sliderTicks}
          value={clamp(sliderValue, min, sliderMax)}
          valueLabel={enabled ? formatDecimalInput(value, step) : nullLabel}
          onChange={updateFromSlider}
        />
        <FieldInput
          aria-label={label}
          className="h-[34px] indent-0 tabular-nums"
          disabled={!enabled}
          max={max}
          min={min}
          step={step}
          {...editableNumberInputProps("decimal")}
          value={rawValue}
          onBlur={commitValue}
          onChange={(event) => changeValue(event.target.value)}
          onKeyDown={blurOnEnter}
        />
      </div>
    </FieldShell>
  );
}
