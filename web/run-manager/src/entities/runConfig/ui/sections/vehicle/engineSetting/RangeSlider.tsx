// web/run-manager/src/entities/runConfig/ui/sections/vehicle/engineSetting/RangeSlider.tsx
import { useCallback, useEffect, useRef, useState } from "react";
import {
  sliderRatio,
  valueFromClientX,
} from "@/entities/runConfig/ui/sections/vehicle/engineSetting/math";
import {
  rangeFillStyle,
  thumbStyle,
  tickStyle,
} from "@/entities/runConfig/ui/sections/vehicle/engineSetting/styles";
import type {
  RangeHandle,
  SliderTick,
} from "@/entities/runConfig/ui/sections/vehicle/engineSetting/types";
import { engineSliderStepPercentLabel } from "@/shared/domain/engineBuckets";
import { cn } from "@/shared/ui/cn";
import { clamp } from "@/shared/ui/configFields/format";

interface RangeSliderProps {
  disabled?: boolean;
  label: string;
  max: number;
  min: number;
  step: number;
  ticks: readonly SliderTick[];
  valueMax: number;
  valueMin: number;
  onChange: (value: { min: number; max: number }) => void;
}

export function RangeSlider({
  disabled = false,
  label,
  max,
  min,
  step,
  ticks,
  valueMax,
  valueMin,
  onChange,
}: RangeSliderProps) {
  const trackRef = useRef<HTMLDivElement>(null);
  const [dragging, setDragging] = useState<RangeHandle | null>(null);
  const minRatio = sliderRatio(valueMin, min, max);
  const maxRatio = sliderRatio(valueMax, min, max);

  const updateHandle = useCallback(
    (handle: RangeHandle, nextValue: number) => {
      if (handle === "min") {
        onChange({ max: valueMax, min: Math.min(nextValue, valueMax) });
        return;
      }
      onChange({ max: Math.max(nextValue, valueMin), min: valueMin });
    },
    [onChange, valueMax, valueMin],
  );

  useEffect(() => {
    if (dragging === null) {
      return undefined;
    }
    const activeHandle = dragging;

    function handlePointerMove(event: PointerEvent) {
      updateHandle(activeHandle, valueFromClientX(trackRef.current, event.clientX, min, max, step));
    }

    function handlePointerUp() {
      setDragging(null);
    }

    window.addEventListener("pointermove", handlePointerMove);
    window.addEventListener("pointerup", handlePointerUp);
    return () => {
      window.removeEventListener("pointermove", handlePointerMove);
      window.removeEventListener("pointerup", handlePointerUp);
    };
  }, [dragging, max, min, step, updateHandle]);

  return (
    <div
      className={cn("vehicle-engine-slider", disabled && "pointer-events-none opacity-60")}
      ref={trackRef}
      onPointerDown={(event) => {
        if (disabled) {
          return;
        }
        const nextValue = valueFromClientX(trackRef.current, event.clientX, min, max, step);
        const handle =
          Math.abs(nextValue - valueMin) <= Math.abs(nextValue - valueMax) ? "min" : "max";
        updateHandle(handle, nextValue);
        setDragging(handle);
      }}
    >
      <div aria-hidden="true" className="vehicle-engine-slider-rail" />
      <div
        aria-hidden="true"
        className="vehicle-engine-slider-fill"
        style={rangeFillStyle(minRatio, maxRatio)}
      />
      <RangeThumb
        disabled={disabled}
        label={`${label} minimum`}
        max={valueMax}
        min={min}
        ratio={minRatio}
        sliding={dragging === "min"}
        step={step}
        value={valueMin}
        onChange={(nextValue) => updateHandle("min", clamp(nextValue, min, valueMax))}
        onPointerDown={() => setDragging("min")}
      />
      <RangeThumb
        disabled={disabled}
        label={`${label} maximum`}
        max={max}
        min={valueMin}
        ratio={maxRatio}
        sliding={dragging === "max"}
        step={step}
        value={valueMax}
        onChange={(nextValue) => updateHandle("max", clamp(nextValue, valueMin, max))}
        onPointerDown={() => setDragging("max")}
      />
      <div className="vehicle-engine-slider-ticks" aria-hidden="true">
        {ticks.map((tick) => (
          <span
            data-label={tick.label}
            key={`${tick.value}-${tick.label}`}
            style={tickStyle(sliderRatio(tick.value, min, max))}
          />
        ))}
      </div>
    </div>
  );
}

interface RangeThumbProps {
  disabled: boolean;
  label: string;
  max: number;
  min: number;
  ratio: number;
  sliding: boolean;
  step: number;
  value: number;
  onChange: (value: number) => void;
  onPointerDown: () => void;
}

function RangeThumb({
  disabled,
  label,
  max,
  min,
  ratio,
  sliding,
  step,
  value,
  onChange,
  onPointerDown,
}: RangeThumbProps) {
  return (
    <button
      aria-label={label}
      aria-valuemax={max}
      aria-valuemin={min}
      aria-valuenow={value}
      aria-valuetext={engineSliderStepPercentLabel(value)}
      className={sliding ? "vehicle-engine-slider-thumb sliding" : "vehicle-engine-slider-thumb"}
      disabled={disabled}
      role="slider"
      style={thumbStyle(ratio)}
      type="button"
      onKeyDown={(event) => {
        if (event.key === "ArrowLeft" || event.key === "ArrowDown") {
          event.preventDefault();
          onChange(value - step);
        }
        if (event.key === "ArrowRight" || event.key === "ArrowUp") {
          event.preventDefault();
          onChange(value + step);
        }
      }}
      onPointerDown={(event) => {
        event.preventDefault();
        event.stopPropagation();
        onPointerDown();
      }}
    />
  );
}
