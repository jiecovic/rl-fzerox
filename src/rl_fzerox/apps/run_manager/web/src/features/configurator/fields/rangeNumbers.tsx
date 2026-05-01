import { useEffect, useState } from "react";

import {
  clamp,
  formatCompactDecimal,
  formatInteger,
  roundSignificant,
} from "@/features/configurator/fields/format";
import { FieldLabel } from "@/features/configurator/fields/label";
import { resetHandler } from "@/features/configurator/fields/reset";
import {
  discreteSliderTicks,
  nearestIndex,
  nearestOption,
  Slider,
} from "@/features/configurator/fields/slider";
import type { SliderTick } from "@/features/configurator/fields/types";

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
  return (
    <div className="field-shell range-field">
      <FieldLabel help={help} label={label} onReset={resetHandler(value, resetValue, onChange)} />
      <div className="range-row">
        <Slider
          ariaLabel={`${label} slider`}
          max={max}
          min={min}
          step={rangeStep}
          ticks={ticks}
          value={clamp(value, min, max)}
          valueLabel={String(value)}
          onChange={onChange}
        />
        <input
          aria-label={label}
          max={max}
          min={min}
          step={numberStep}
          type="number"
          value={value}
          onChange={(event) => onChange(Number(event.target.value))}
        />
      </div>
    </div>
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
  const [rawValue, setRawValue] = useState(formatInteger(value));

  useEffect(() => {
    setRawValue(formatInteger(value));
  }, [value]);

  function commitValue() {
    const parsed = Number(rawValue.replace(/[,_\s]/g, ""));
    if (!Number.isSafeInteger(parsed) || parsed < min || parsed > max) {
      setRawValue(formatInteger(value));
      return;
    }
    onChange(parsed);
    setRawValue(formatInteger(parsed));
  }

  function updateFromSlider(nextValue: number) {
    const rounded = Math.round(nextValue);
    onChange(rounded);
    setRawValue(formatInteger(rounded));
  }

  return (
    <div className="field-shell range-field">
      <FieldLabel
        help={help}
        label={label}
        onReset={resetHandler(value, resetValue, updateFromSlider)}
      />
      <div className="range-row">
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
        <input
          aria-label={label}
          inputMode="numeric"
          spellCheck={false}
          value={rawValue}
          onBlur={commitValue}
          onChange={(event) => setRawValue(event.target.value)}
        />
      </div>
    </div>
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
    const parsed = Number(rawValue);
    if (!Number.isSafeInteger(parsed) || parsed < minManual || parsed > maxManual) {
      setRawValue(String(value));
      return;
    }
    const committed = snapManualToOptions ? nearestOption(parsed, sliderValues) : parsed;
    onChange(committed);
    setRawValue(String(committed));
  }

  function changeManualValue(nextRawValue: string) {
    setRawValue(nextRawValue);
    if (!snapManualToOptions) {
      return;
    }
    const parsed = Number(nextRawValue);
    if (Number.isSafeInteger(parsed) && parsed >= minManual && parsed <= maxManual) {
      onChange(nearestOption(parsed, sliderValues));
    }
  }

  function updateFromSlider(index: number) {
    const nextValue = sliderValues[index] ?? value;
    onChange(nextValue);
    setRawValue(String(nextValue));
  }

  return (
    <div className="field-shell range-field">
      <FieldLabel help={help} label={label} onReset={resetHandler(value, resetValue, onChange)} />
      <div className="range-row">
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
        <input
          aria-label={label}
          max={maxManual}
          min={minManual}
          inputMode="numeric"
          spellCheck={false}
          step="1"
          type="number"
          value={rawValue}
          onBlur={commitManualValue}
          onChange={(event) => changeManualValue(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter") {
              event.currentTarget.blur();
            }
          }}
        />
      </div>
    </div>
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
  const [rawValue, setRawValue] = useState(value.toExponential(2));
  const logMin = Math.log10(min);
  const logMax = Math.log10(max);
  const logValue = clamp(Math.log10(value), logMin, logMax);

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

  function updateFromSlider(nextLogValue: number) {
    onChange(roundSignificant(10 ** nextLogValue, 3));
  }

  return (
    <div className="field-shell range-field">
      <FieldLabel help={help} label={label} onReset={resetHandler(value, resetValue, onChange)} />
      <div className="range-row">
        <Slider
          ariaLabel={`${label} slider`}
          max={logMax}
          min={logMin}
          step={0.01}
          ticks={ticks.map((tick) => ({ label: tick.label, value: Math.log10(tick.value) }))}
          value={logValue}
          valueLabel={rawValue}
          onChange={updateFromSlider}
        />
        <input
          aria-label={label}
          inputMode="decimal"
          spellCheck={false}
          value={rawValue}
          onBlur={commitValue}
          onChange={(event) => setRawValue(event.target.value)}
        />
      </div>
    </div>
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
}) {
  const enabled = value !== null;
  const sliderValue = enabled ? value : defaultValue;

  return (
    <div className="field-shell optional-number-field">
      <FieldLabel help={help} label={label} onReset={resetHandler(value, resetValue, onChange)} />
      <div className={enabled ? "optional-number-row" : "optional-number-row disabled"}>
        <button
          aria-label={`${label} enabled`}
          aria-pressed={enabled}
          className={enabled ? "switch-button active" : "switch-button"}
          type="button"
          onClick={() => onChange(enabled ? null : defaultValue)}
        >
          <span aria-hidden="true" />
          <strong>{enabled ? "On" : "Off"}</strong>
        </button>
        <Slider
          ariaLabel={`${label} slider`}
          disabled={!enabled}
          max={max}
          min={min}
          step={Number(step)}
          ticks={[
            { value: min, label: formatCompactDecimal(min) },
            { value: max, label: formatCompactDecimal(max) },
          ]}
          value={clamp(sliderValue, min, max)}
          valueLabel={String(enabled ? value : defaultValue)}
          onChange={onChange}
        />
        <input
          aria-label={label}
          disabled={!enabled}
          max={max}
          min={min}
          step={step}
          type="number"
          value={enabled ? value : defaultValue}
          onChange={(event) => onChange(Number(event.target.value))}
        />
      </div>
    </div>
  );
}
