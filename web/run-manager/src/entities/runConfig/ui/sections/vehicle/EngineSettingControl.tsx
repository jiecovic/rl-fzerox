// web/run-manager/src/entities/runConfig/ui/sections/vehicle/EngineSettingControl.tsx

import { RangeSlider } from "@/entities/runConfig/ui/sections/vehicle/engineSetting/RangeSlider";
import { SingleSlider } from "@/entities/runConfig/ui/sections/vehicle/engineSetting/SingleSlider";
import type {
  EngineMode,
  SliderTick,
} from "@/entities/runConfig/ui/sections/vehicle/engineSetting/types";
import { engineSliderStepPercentLabel } from "@/shared/domain/engineBuckets";
import { FieldLabel } from "@/shared/ui/configFields/label";
import { FieldShell } from "@/shared/ui/Field";

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
    <FieldShell>
      <FieldLabel help={help} label={label} onReset={resetHandler} />
      <div className="grid grid-cols-[minmax(0,1fr)_150px] items-center gap-3 max-[720px]:grid-cols-1">
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
        <div className="w-[150px] max-[720px]:w-full">
          {mode === "fixed" ? (
            <EngineStepReadout value={fixedValue} />
          ) : (
            <EngineRangeReadout max={rangeMax} min={rangeMin} />
          )}
        </div>
      </div>
    </FieldShell>
  );
}

function EngineStepReadout({ value }: { value: number }) {
  return (
    <div className="grid justify-end text-right tabular-nums max-[720px]:justify-start max-[720px]:text-left">
      <div className="text-lg font-semibold leading-none text-app-text">
        {engineSliderStepPercentLabel(value)}
      </div>
    </div>
  );
}

function EngineRangeReadout({ max, min }: { max: number; min: number }) {
  return (
    <div className="grid justify-end text-right tabular-nums max-[720px]:justify-start max-[720px]:text-left">
      <div className="text-lg font-semibold leading-none text-app-text">
        {engineSliderStepPercentLabel(min)}-{engineSliderStepPercentLabel(max)}
      </div>
    </div>
  );
}
