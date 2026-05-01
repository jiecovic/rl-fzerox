import type { CSSProperties } from "react";

import { clamp, formatCompactNumber } from "@/features/configurator/fields/format";
import type { SliderTick } from "@/features/configurator/fields/types";

export function Slider({
  ariaLabel,
  disabled = false,
  max,
  min,
  step,
  ticks,
  value,
  onChange,
}: {
  ariaLabel: string;
  disabled?: boolean;
  max: number;
  min: number;
  step: number;
  ticks: readonly SliderTick[];
  value: number;
  onChange: (value: number) => void;
}) {
  return (
    <div className="slider-control">
      <input
        aria-label={ariaLabel}
        disabled={disabled}
        max={max}
        min={min}
        step={step}
        type="range"
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
      />
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
  const percent = ((value - min) / (max - min)) * 100;
  return { left: `${clamp(percent, 0, 100)}%` };
}
