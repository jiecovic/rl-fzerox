// src/rl_fzerox/apps/run_manager/web/src/features/configurator/sections/vehicle/EngineSettingControl.tsx
import { FieldLabel } from "@/features/configurator/fields/label";
import {
  blurOnEnter,
  editableNumberInputProps,
  parseSafeIntegerInput,
  useEditableNumberInput,
} from "@/features/configurator/fields/numberInput";
import { RangeSlider } from "@/features/configurator/sections/vehicle/engineSetting/RangeSlider";
import { SingleSlider } from "@/features/configurator/sections/vehicle/engineSetting/SingleSlider";
import type {
  EngineMode,
  SliderTick,
} from "@/features/configurator/sections/vehicle/engineSetting/types";
import { FieldInput, FieldShell } from "@/shared/ui/Field";

interface EngineSettingControlProps {
  defaultFixedValue: number;
  defaultRangeMax: number;
  defaultRangeMin: number;
  fixedValue: number;
  help: string;
  label: string;
  max: number;
  min: number;
  mode: EngineMode;
  rangeMax: number;
  rangeMin: number;
  ticks: readonly SliderTick[];
  onFixedChange: (value: number) => void;
  onRangeChange: (value: { min: number; max: number }) => void;
}

export function EngineSettingControl({
  defaultFixedValue,
  defaultRangeMax,
  defaultRangeMin,
  fixedValue,
  help,
  label,
  max,
  min,
  mode,
  rangeMax,
  rangeMin,
  ticks,
  onFixedChange,
  onRangeChange,
}: EngineSettingControlProps) {
  const resetHandler =
    mode === "fixed"
      ? fixedValue === defaultFixedValue
        ? undefined
        : () => onFixedChange(defaultFixedValue)
      : rangeMin === defaultRangeMin && rangeMax === defaultRangeMax
        ? undefined
        : () => onRangeChange({ max: defaultRangeMax, min: defaultRangeMin });
  const fixedInput = useEditableNumberInput({
    format: String,
    formattedValue: String(fixedValue),
    normalize: (nextValue) => clampInteger(nextValue, min, max),
    onCommit: onFixedChange,
    parse: (rawValue) => parseSafeIntegerInput(rawValue, { max, min }),
  });
  const rangeMinInput = useEditableNumberInput({
    format: String,
    formattedValue: String(rangeMin),
    normalize: (nextValue) => clampInteger(nextValue, min, rangeMax),
    onCommit: (nextMin) => onRangeChange({ max: rangeMax, min: nextMin }),
    parse: (rawValue) => parseSafeIntegerInput(rawValue, { max: rangeMax, min }),
  });
  const rangeMaxInput = useEditableNumberInput({
    format: String,
    formattedValue: String(rangeMax),
    normalize: (nextValue) => clampInteger(nextValue, rangeMin, max),
    onCommit: (nextMax) => onRangeChange({ max: nextMax, min: rangeMin }),
    parse: (rawValue) => parseSafeIntegerInput(rawValue, { max, min: rangeMin }),
  });

  return (
    <FieldShell>
      <FieldLabel help={help} label={label} onReset={resetHandler} />
      <div className="grid grid-cols-[minmax(0,1fr)_176px] items-center gap-3 max-[720px]:grid-cols-1">
        <div className="min-w-0">
          {mode === "fixed" ? (
            <SingleSlider
              label={label}
              max={max}
              min={min}
              step={1}
              ticks={ticks}
              value={fixedValue}
              onChange={onFixedChange}
            />
          ) : (
            <RangeSlider
              label={label}
              max={max}
              min={min}
              step={1}
              ticks={ticks}
              valueMax={rangeMax}
              valueMin={rangeMin}
              onChange={onRangeChange}
            />
          )}
        </div>
        <div className="w-[176px] max-[720px]:w-full">
          {mode === "fixed" ? (
            <div className="grid grid-cols-[repeat(2,84px)] justify-end gap-2 max-[720px]:justify-start">
              <FieldInput
                aria-label={label}
                className="col-span-2 !h-[34px] !w-[176px] text-center tabular-nums"
                max={max}
                min={min}
                step={1}
                {...editableNumberInputProps("integer")}
                value={fixedInput.rawValue}
                onBlur={fixedInput.commitRawValue}
                onChange={(event) => fixedInput.changeRawValue(event.target.value)}
                onKeyDown={blurOnEnter}
              />
            </div>
          ) : (
            <div className="grid grid-cols-[repeat(2,84px)] justify-end gap-2 max-[720px]:justify-start">
              <FieldInput
                aria-label={`${label} minimum`}
                className="!h-[34px] !w-[84px] text-center tabular-nums"
                max={rangeMax}
                min={min}
                step={1}
                {...editableNumberInputProps("integer")}
                value={rangeMinInput.rawValue}
                onBlur={rangeMinInput.commitRawValue}
                onChange={(event) => rangeMinInput.changeRawValue(event.target.value)}
                onKeyDown={blurOnEnter}
              />
              <FieldInput
                aria-label={`${label} maximum`}
                className="!h-[34px] !w-[84px] text-center tabular-nums"
                max={max}
                min={rangeMin}
                step={1}
                {...editableNumberInputProps("integer")}
                value={rangeMaxInput.rawValue}
                onBlur={rangeMaxInput.commitRawValue}
                onChange={(event) => rangeMaxInput.changeRawValue(event.target.value)}
                onKeyDown={blurOnEnter}
              />
            </div>
          )}
        </div>
      </div>
    </FieldShell>
  );
}

function clampInteger(value: number, min: number, max: number) {
  return Math.min(Math.max(Math.round(value), min), max);
}
