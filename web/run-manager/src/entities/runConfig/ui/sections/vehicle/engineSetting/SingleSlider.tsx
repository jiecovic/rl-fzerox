// web/run-manager/src/entities/runConfig/ui/sections/vehicle/engineSetting/SingleSlider.tsx
import { useEffect, useRef, useState } from "react";
import {
  sliderRatio,
  valueFromClientX,
} from "@/entities/runConfig/ui/sections/vehicle/engineSetting/math";
import {
  singleFillStyle,
  thumbStyle,
  tickStyle,
} from "@/entities/runConfig/ui/sections/vehicle/engineSetting/styles";
import type { SliderTick } from "@/entities/runConfig/ui/sections/vehicle/engineSetting/types";
import { engineSliderStepPercentLabel } from "@/shared/domain/engineBuckets";
import { IconButton } from "@/shared/ui/Button";
import { cn } from "@/shared/ui/cn";
import { clamp } from "@/shared/ui/configFields/format";
import { ChevronIcon } from "@/shared/ui/icons";

interface SingleSliderProps {
  disabled?: boolean;
  label: string;
  max: number;
  min: number;
  step: number;
  ticks: readonly SliderTick[];
  value: number;
  onChange: (value: number) => void;
}

export function SingleSlider({
  disabled = false,
  label,
  max,
  min,
  step,
  ticks,
  value,
  onChange,
}: SingleSliderProps) {
  const trackRef = useRef<HTMLDivElement>(null);
  const [dragging, setDragging] = useState(false);
  const ratio = sliderRatio(value, min, max);
  const decreaseValue = () => onChange(clamp(value - step, min, max));
  const increaseValue = () => onChange(clamp(value + step, min, max));

  useEffect(() => {
    if (!dragging) {
      return undefined;
    }

    function handlePointerMove(event: PointerEvent) {
      onChange(valueFromClientX(trackRef.current, event.clientX, min, max, step));
    }

    function handlePointerUp() {
      setDragging(false);
    }

    window.addEventListener("pointermove", handlePointerMove);
    window.addEventListener("pointerup", handlePointerUp);
    return () => {
      window.removeEventListener("pointermove", handlePointerMove);
      window.removeEventListener("pointerup", handlePointerUp);
    };
  }, [dragging, max, min, onChange, step]);

  return (
    <div className="grid grid-cols-[28px_minmax(0,1fr)_28px] items-start gap-2">
      <IconButton
        aria-label={`Decrease ${label}`}
        className="mt-px"
        disabled={disabled || value <= min}
        size="small"
        onClick={decreaseValue}
      >
        <span className="rotate-180">
          <ChevronIcon />
        </span>
      </IconButton>
      <div
        className={cn("vehicle-engine-slider", disabled && "pointer-events-none opacity-60")}
        ref={trackRef}
        onPointerDown={(event) => {
          if (disabled) {
            return;
          }
          onChange(valueFromClientX(trackRef.current, event.clientX, min, max, step));
          setDragging(true);
        }}
      >
        <div aria-hidden="true" className="vehicle-engine-slider-rail" />
        <div
          aria-hidden="true"
          className="vehicle-engine-slider-fill"
          style={singleFillStyle(ratio)}
        />
        <button
          aria-label={`${label} slider`}
          aria-valuemax={max}
          aria-valuemin={min}
          aria-valuenow={value}
          aria-valuetext={engineSliderStepPercentLabel(value)}
          className={
            dragging ? "vehicle-engine-slider-thumb sliding" : "vehicle-engine-slider-thumb"
          }
          disabled={disabled}
          role="slider"
          style={thumbStyle(ratio)}
          type="button"
          onKeyDown={(event) => {
            if (event.key === "ArrowLeft" || event.key === "ArrowDown") {
              event.preventDefault();
              decreaseValue();
            }
            if (event.key === "ArrowRight" || event.key === "ArrowUp") {
              event.preventDefault();
              increaseValue();
            }
          }}
          onPointerDown={(event) => {
            event.preventDefault();
            event.stopPropagation();
            setDragging(true);
          }}
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
      <IconButton
        aria-label={`Increase ${label}`}
        className="mt-px"
        disabled={disabled || value >= max}
        size="small"
        onClick={increaseValue}
      >
        <ChevronIcon />
      </IconButton>
    </div>
  );
}
