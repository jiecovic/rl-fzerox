// src/rl_fzerox/apps/run_manager/web/src/features/configurator/sections/vehicle/engineSetting/SingleSlider.tsx
import { useEffect, useRef, useState } from "react";

import { clamp } from "@/features/configurator/fields/format";

import { sliderRatio, valueFromClientX } from "./math";
import { singleFillStyle, thumbStyle, tickStyle } from "./styles";
import type { SliderTick } from "./types";

interface SingleSliderProps {
  label: string;
  max: number;
  min: number;
  step: number;
  ticks: readonly SliderTick[];
  value: number;
  onChange: (value: number) => void;
}

export function SingleSlider({ label, max, min, step, ticks, value, onChange }: SingleSliderProps) {
  const trackRef = useRef<HTMLDivElement>(null);
  const [dragging, setDragging] = useState(false);
  const ratio = sliderRatio(value, min, max);

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
    <div
      className="vehicle-engine-slider"
      ref={trackRef}
      onPointerDown={(event) => {
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
        className={dragging ? "vehicle-engine-slider-thumb sliding" : "vehicle-engine-slider-thumb"}
        role="slider"
        style={thumbStyle(ratio)}
        type="button"
        onKeyDown={(event) => {
          if (event.key === "ArrowLeft" || event.key === "ArrowDown") {
            event.preventDefault();
            onChange(clamp(value - step, min, max));
          }
          if (event.key === "ArrowRight" || event.key === "ArrowUp") {
            event.preventDefault();
            onChange(clamp(value + step, min, max));
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
  );
}
