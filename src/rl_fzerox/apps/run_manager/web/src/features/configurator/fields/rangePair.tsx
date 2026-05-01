import { useCallback, useEffect, useRef, useState } from "react";

import {
  clamp,
  formatCompactDecimal,
  formatCompactNumber,
} from "@/features/configurator/fields/format";
import { FieldLabel } from "@/features/configurator/fields/label";

type RangeHandle = "min" | "max";

export function OptionalRangePairField({
  help,
  label,
  valueMin,
  valueMax,
  onChange,
  defaultMin,
  defaultMax,
  max,
  min,
  resetMin,
  resetMax,
  step = 1,
}: {
  help: string;
  label: string;
  valueMin: number | null;
  valueMax: number | null;
  onChange: (value: { min: number | null; max: number | null }) => void;
  defaultMin: number;
  defaultMax: number;
  max: number;
  min: number;
  resetMin?: number | null;
  resetMax?: number | null;
  step?: number;
}) {
  const enabled = valueMin !== null && valueMax !== null;
  const activeMin = enabled ? valueMin : defaultMin;
  const activeMax = enabled ? valueMax : defaultMax;
  const fieldResetHandler =
    resetMin === undefined ||
    resetMax === undefined ||
    (Object.is(valueMin, resetMin) && Object.is(valueMax, resetMax))
      ? undefined
      : () => onChange({ min: resetMin, max: resetMax });

  function updateEnabled() {
    onChange(enabled ? { min: null, max: null } : { min: defaultMin, max: defaultMax });
  }

  function updateMin(nextValue: number) {
    if (enabled) {
      onChange({ min: clamp(snapToStep(nextValue, min, step), min, activeMax), max: activeMax });
    }
  }

  function updateMax(nextValue: number) {
    if (enabled) {
      onChange({ min: activeMin, max: clamp(snapToStep(nextValue, min, step), activeMin, max) });
    }
  }

  return (
    <div className="field-shell range-pair-field">
      <FieldLabel help={help} label={label} onReset={fieldResetHandler} />
      <div className={enabled ? "range-pair-row" : "range-pair-row disabled"}>
        <button
          aria-label={`${label} enabled`}
          aria-pressed={enabled}
          className={enabled ? "switch-button active" : "switch-button"}
          type="button"
          onClick={updateEnabled}
        >
          <span aria-hidden="true" />
          <strong>{enabled ? "On" : "Off"}</strong>
        </button>
        <RangePairSlider
          disabled={!enabled}
          label={label}
          max={max}
          min={min}
          step={step}
          valueMax={activeMax}
          valueMin={activeMin}
          onMaxChange={updateMax}
          onMinChange={updateMin}
        />
        <div className="range-pair-values">
          <input
            aria-label={`${label} minimum`}
            disabled={!enabled}
            max={activeMax}
            min={min}
            step={step}
            type="number"
            value={activeMin}
            onChange={(event) => updateMin(Number(event.target.value))}
          />
          <input
            aria-label={`${label} maximum`}
            disabled={!enabled}
            max={max}
            min={activeMin}
            step={step}
            type="number"
            value={activeMax}
            onChange={(event) => updateMax(Number(event.target.value))}
          />
        </div>
      </div>
    </div>
  );
}

function RangePairSlider({
  disabled,
  label,
  max,
  min,
  step,
  valueMax,
  valueMin,
  onMaxChange,
  onMinChange,
}: {
  disabled: boolean;
  label: string;
  max: number;
  min: number;
  step: number;
  valueMax: number;
  valueMin: number;
  onMaxChange: (value: number) => void;
  onMinChange: (value: number) => void;
}) {
  const trackRef = useRef<HTMLDivElement>(null);
  const [dragging, setDragging] = useState<RangeHandle | null>(null);
  const lowerPercent = rangePercent(valueMin, min, max);
  const upperPercent = rangePercent(valueMax, min, max);

  const updateHandle = useCallback(
    (handle: RangeHandle, nextValue: number) => {
      if (handle === "min") {
        onMinChange(nextValue);
        return;
      }
      onMaxChange(nextValue);
    },
    [onMaxChange, onMinChange],
  );

  useEffect(() => {
    if (dragging === null || disabled) {
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
  }, [disabled, dragging, max, min, step, updateHandle]);

  function beginDrag(handle: RangeHandle) {
    if (!disabled) {
      setDragging(handle);
    }
  }

  function handleTrackPointerDown(event: React.PointerEvent<HTMLDivElement>) {
    if (disabled) {
      return;
    }
    const nextValue = valueFromClientX(trackRef.current, event.clientX, min, max, step);
    const handle = Math.abs(nextValue - valueMin) <= Math.abs(nextValue - valueMax) ? "min" : "max";
    updateHandle(handle, nextValue);
    setDragging(handle);
  }

  return (
    <div className="range-pair-slider" ref={trackRef} onPointerDown={handleTrackPointerDown}>
      <div aria-hidden="true" className="range-pair-rail" />
      <div
        aria-hidden="true"
        className="range-pair-fill"
        style={{ left: `${lowerPercent}%`, right: `${100 - upperPercent}%` }}
      />
      <RangeThumb
        disabled={disabled}
        label={`${label} minimum`}
        max={valueMax}
        min={min}
        percent={lowerPercent}
        step={step}
        value={valueMin}
        onChange={onMinChange}
        sliding={dragging === "min"}
        onPointerDown={() => beginDrag("min")}
      />
      <RangeThumb
        disabled={disabled}
        label={`${label} maximum`}
        max={max}
        min={valueMin}
        percent={upperPercent}
        step={step}
        value={valueMax}
        onChange={onMaxChange}
        sliding={dragging === "max"}
        onPointerDown={() => beginDrag("max")}
      />
      <div className="slider-ticks" aria-hidden="true">
        <span data-label={formatCompactDecimal(min)} style={{ left: "0%" }} />
        <span data-label={formatCompactDecimal(max)} style={{ left: "100%" }} />
      </div>
    </div>
  );
}

function RangeThumb({
  disabled,
  label,
  max,
  min,
  percent,
  step,
  value,
  onChange,
  sliding,
  onPointerDown,
}: {
  disabled: boolean;
  label: string;
  max: number;
  min: number;
  percent: number;
  step: number;
  value: number;
  onChange: (value: number) => void;
  sliding: boolean;
  onPointerDown: () => void;
}) {
  return (
    <button
      aria-label={label}
      aria-valuemax={max}
      aria-valuemin={min}
      aria-valuenow={value}
      className={sliding ? "range-pair-thumb sliding" : "range-pair-thumb"}
      disabled={disabled}
      role="slider"
      style={{ left: `${percent}%` }}
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
    >
      <span className="slider-value-bubble" aria-hidden="true">
        {formatCompactNumber(value)}
      </span>
    </button>
  );
}

function valueFromClientX(
  track: HTMLDivElement | null,
  clientX: number,
  min: number,
  max: number,
  step: number,
) {
  if (track === null) {
    return min;
  }
  const rect = track.getBoundingClientRect();
  const ratio = clamp((clientX - rect.left) / rect.width, 0, 1);
  return snapToStep(min + ratio * (max - min), min, step);
}

function rangePercent(value: number, min: number, max: number) {
  return clamp(((value - min) / (max - min)) * 100, 0, 100);
}

function snapToStep(value: number, min: number, step: number) {
  const snapped = min + Math.round((value - min) / step) * step;
  const decimals = step.toString().split(".")[1]?.length ?? 0;
  return Number(snapped.toFixed(decimals));
}
