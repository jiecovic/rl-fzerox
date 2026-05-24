// src/rl_fzerox/apps/run_manager/web/src/features/configurator/sections/vehicle/EngineSettingControl.tsx
import { FieldLabel } from "@/features/configurator/fields/label";

import { parseBoundedInt } from "@/features/configurator/sections/vehicle/engineSetting/math";
import { RangeSlider } from "@/features/configurator/sections/vehicle/engineSetting/RangeSlider";
import { SingleSlider } from "@/features/configurator/sections/vehicle/engineSetting/SingleSlider";
import type {
  EngineMode,
  SliderTick,
} from "@/features/configurator/sections/vehicle/engineSetting/types";

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

  return (
    <div className="field-shell vehicle-engine-control">
      <FieldLabel help={help} label={label} onReset={resetHandler} />
      <div className="vehicle-engine-control-row">
        <div className="vehicle-engine-slider-slot">
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
        <div className="vehicle-engine-value-slot">
          {mode === "fixed" ? (
            <div className="vehicle-engine-value-shell">
              <input
                aria-label={label}
                className="vehicle-engine-value-single"
                inputMode="numeric"
                max={max}
                min={min}
                spellCheck={false}
                step={1}
                type="number"
                value={fixedValue}
                onChange={(event) => onFixedChange(parseBoundedInt(event.target.value, min, max))}
              />
            </div>
          ) : (
            <div className="vehicle-engine-value-shell">
              <input
                aria-label={`${label} minimum`}
                inputMode="numeric"
                max={rangeMax}
                min={min}
                spellCheck={false}
                step={1}
                type="number"
                value={rangeMin}
                onChange={(event) =>
                  onRangeChange({
                    max: rangeMax,
                    min: Math.min(parseBoundedInt(event.target.value, min, max), rangeMax),
                  })
                }
              />
              <input
                aria-label={`${label} maximum`}
                inputMode="numeric"
                max={max}
                min={rangeMin}
                spellCheck={false}
                step={1}
                type="number"
                value={rangeMax}
                onChange={(event) =>
                  onRangeChange({
                    max: Math.max(parseBoundedInt(event.target.value, min, max), rangeMin),
                    min: rangeMin,
                  })
                }
              />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
