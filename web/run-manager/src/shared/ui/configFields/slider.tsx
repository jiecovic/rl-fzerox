// web/run-manager/src/shared/ui/configFields/slider.tsx
import type { CSSProperties } from "react";
import { useState } from "react";

import { clamp, formatCompactDecimal, formatCompactNumber } from "@/shared/ui/configFields/format";
import type { SliderTick } from "@/shared/ui/configFields/types";

export function Slider({
  ariaLabel,
  disabled = false,
  max,
  min,
  step,
  ticks,
  value,
  valueLabel,
  onChange,
}: {
  ariaLabel: string;
  disabled?: boolean;
  max: number;
  min: number;
  step: number;
  ticks: readonly SliderTick[];
  value: number;
  valueLabel?: string;
  onChange: (value: number) => void;
}) {
  const [sliding, setSliding] = useState(false);
  const valueRatio = tickRatio(value, min, max);

  return (
    <div className={sliding ? "slider-control sliding" : "slider-control"}>
      <div className="slider-track-shell">
        <div className="slider-track-content">
          <input
            aria-label={ariaLabel}
            disabled={disabled}
            max={max}
            min={min}
            step={step}
            type="range"
            value={value}
            onChange={(event) => onChange(Number(event.target.value))}
            onPointerDown={() => setSliding(true)}
            onPointerUp={() => setSliding(false)}
            onPointerCancel={() => setSliding(false)}
            onBlur={() => setSliding(false)}
          />
          <span
            className="slider-value-bubble"
            style={sliderBubbleStyle(valueRatio)}
            aria-hidden="true"
          >
            {valueLabel ?? formatCompactDecimal(value)}
          </span>
          {ticks.length > 0 ? (
            <div className="slider-ticks" aria-hidden="true">
              {ticks.map((tick) => (
                <span
                  data-label={tick.label}
                  key={`${tick.value}-${tick.label}`}
                  style={tickStyle(tick.value, min, max)}
                />
              ))}
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
}

export function discreteSliderTicks(options: readonly number[]) {
  if (options.length <= 5) {
    return options.map((option, index) => ({ label: formatCompactNumber(option), value: index }));
  }
  const middle = Math.floor((options.length - 1) / 2);
  return [0, middle, options.length - 1].map((index) => ({
    label: formatCompactNumber(options[index] ?? 0),
    value: index,
  }));
}

export function nearestIndex(value: number, options: readonly number[]) {
  let nearest = 0;
  let nearestDistance = Number.POSITIVE_INFINITY;
  options.forEach((option, index) => {
    const distance = Math.abs(option - value);
    if (distance < nearestDistance) {
      nearest = index;
      nearestDistance = distance;
    }
  });
  return nearest;
}

export function nearestOption(value: number, options: readonly number[]) {
  return options[nearestIndex(value, options)] ?? value;
}

function tickStyle(value: number, min: number, max: number): CSSProperties {
  return { "--slider-tick-ratio": `${tickRatio(value, min, max)}` } as CSSProperties;
}

function sliderBubbleStyle(valueRatio: number): CSSProperties {
  return { "--slider-value-ratio": `${valueRatio}` } as CSSProperties;
}

function tickRatio(value: number, min: number, max: number) {
  const ratio = (value - min) / (max - min);
  return clamp(ratio, 0, 1);
}
