// src/rl_fzerox/apps/run_manager/web/src/features/configurator/fields/rangePair.tsx
import type { CSSProperties } from "react";
import { useCallback, useEffect, useRef, useState } from "react";

import {
  clamp,
  formatCompactDecimal,
  formatCompactNumber,
} from "@/features/configurator/fields/format";
import { FieldLabel } from "@/features/configurator/fields/label";
import type { SliderTick } from "@/features/configurator/fields/types";

type RangeHandle = "min" | "max";

interface RangePairFieldProps {
  help: string;
  label: string;
  max: number;
  min: number;
  onChange: (value: { min: number; max: number }) => void;
  resetMax?: number;
  resetMin?: number;
  step?: number;
  ticks?: readonly SliderTick[];
  valueMax: number;
  valueMin: number;
}

export function RangePairField({
  help,
  label,
  max,
  min,
  onChange,
  resetMax,
  resetMin,
  step = 1,
  ticks,
  valueMax,
  valueMin,
}: RangePairFieldProps) {
  const [rawMin, setRawMin] = useState(String(valueMin));
  const [rawMax, setRawMax] = useState(String(valueMax));
  const fieldResetHandler =
    resetMin === undefined ||
    resetMax === undefined ||
    (Object.is(valueMin, resetMin) && Object.is(valueMax, resetMax))
      ? undefined
      : () => onChange({ min: resetMin, max: resetMax });

  useEffect(() => {
    setRawMin(String(valueMin));
  }, [valueMin]);

  useEffect(() => {
    setRawMax(String(valueMax));
  }, [valueMax]);

  function updateMin(nextValue: number, syncInput = true) {
    const normalized = normalizeRangeValue(nextValue, min, valueMax, step);
    onChange({ min: normalized, max: valueMax });
    if (syncInput) {
      setRawMin(String(normalized));
    }
  }

  function updateMax(nextValue: number, syncInput = true) {
    const normalized = normalizeRangeValue(nextValue, valueMin, max, step);
    onChange({ min: valueMin, max: normalized });
    if (syncInput) {
      setRawMax(String(normalized));
    }
  }

  function changeMin(nextRawValue: string) {
    setRawMin(nextRawValue);
    const parsed = parseEditableNumberInput(nextRawValue);
    if (parsed !== null) {
      updateMin(parsed, false);
    }
  }

  function changeMax(nextRawValue: string) {
    setRawMax(nextRawValue);
    const parsed = parseEditableNumberInput(nextRawValue);
    if (parsed !== null) {
      updateMax(parsed, false);
    }
  }

  function commitMin() {
    const parsed = parseEditableNumberInput(rawMin);
    if (parsed === null) {
      setRawMin(String(valueMin));
      return;
    }
    updateMin(parsed);
  }

  function commitMax() {
    const parsed = parseEditableNumberInput(rawMax);
    if (parsed === null) {
      setRawMax(String(valueMax));
      return;
    }
    updateMax(parsed);
  }

  return (
    <div className="field-shell range-pair-field">
      <FieldLabel help={help} label={label} onReset={fieldResetHandler} />
      <div className="range-pair-row range-pair-row-bare">
        <RangePairSlider
          disabled={false}
          label={label}
          max={max}
          min={min}
          step={step}
          ticks={ticks}
          valueMax={valueMax}
          valueMin={valueMin}
          onMaxChange={updateMax}
          onMinChange={updateMin}
        />
        <div className="range-pair-values">
          <input
            aria-label={`${label} minimum`}
            max={valueMax}
            min={min}
            step={step}
            type="number"
            value={rawMin}
            onBlur={commitMin}
            onChange={(event) => changeMin(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter") {
                event.currentTarget.blur();
              }
            }}
          />
          <input
            aria-label={`${label} maximum`}
            max={max}
            min={valueMin}
            step={step}
            type="number"
            value={rawMax}
            onBlur={commitMax}
            onChange={(event) => changeMax(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter") {
                event.currentTarget.blur();
              }
            }}
          />
        </div>
      </div>
    </div>
  );
}

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
  const [rawMin, setRawMin] = useState(String(activeMin));
  const [rawMax, setRawMax] = useState(String(activeMax));
  const fieldResetHandler =
    resetMin === undefined ||
    resetMax === undefined ||
    (Object.is(valueMin, resetMin) && Object.is(valueMax, resetMax))
      ? undefined
      : () => onChange({ min: resetMin, max: resetMax });

  useEffect(() => {
    setRawMin(String(activeMin));
  }, [activeMin]);

  useEffect(() => {
    setRawMax(String(activeMax));
  }, [activeMax]);

  function updateEnabled() {
    if (enabled) {
      onChange({ min: null, max: null });
      return;
    }
    const normalizedMin = normalizeRangeValue(defaultMin, min, defaultMax, step);
    const normalizedMax = normalizeRangeValue(defaultMax, normalizedMin, max, step);
    onChange({ min: normalizedMin, max: normalizedMax });
    setRawMin(String(normalizedMin));
    setRawMax(String(normalizedMax));
  }

  function updateMin(nextValue: number, syncInput = true) {
    if (enabled) {
      const normalized = normalizeRangeValue(nextValue, min, activeMax, step);
      onChange({ min: normalized, max: activeMax });
      if (syncInput) {
        setRawMin(String(normalized));
      }
    }
  }

  function updateMax(nextValue: number, syncInput = true) {
    if (enabled) {
      const normalized = normalizeRangeValue(nextValue, activeMin, max, step);
      onChange({ min: activeMin, max: normalized });
      if (syncInput) {
        setRawMax(String(normalized));
      }
    }
  }

  function changeMin(nextRawValue: string) {
    setRawMin(nextRawValue);
    const parsed = parseEditableNumberInput(nextRawValue);
    if (parsed !== null) {
      updateMin(parsed, false);
    }
  }

  function changeMax(nextRawValue: string) {
    setRawMax(nextRawValue);
    const parsed = parseEditableNumberInput(nextRawValue);
    if (parsed !== null) {
      updateMax(parsed, false);
    }
  }

  function commitMin() {
    const parsed = parseEditableNumberInput(rawMin);
    if (parsed === null) {
      setRawMin(String(activeMin));
      return;
    }
    updateMin(parsed);
  }

  function commitMax() {
    const parsed = parseEditableNumberInput(rawMax);
    if (parsed === null) {
      setRawMax(String(activeMax));
      return;
    }
    updateMax(parsed);
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
            value={rawMin}
            onBlur={commitMin}
            onChange={(event) => changeMin(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter") {
                event.currentTarget.blur();
              }
            }}
          />
          <input
            aria-label={`${label} maximum`}
            disabled={!enabled}
            max={max}
            min={activeMin}
            step={step}
            type="number"
            value={rawMax}
            onBlur={commitMax}
            onChange={(event) => changeMax(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter") {
                event.currentTarget.blur();
              }
            }}
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
  ticks,
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
  ticks?: readonly SliderTick[];
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

  const sliderTicks = ticks ?? [
    { label: formatCompactDecimal(min), value: min },
    { label: formatCompactDecimal(max), value: max },
  ];

  return (
    <div className="range-pair-slider" ref={trackRef} onPointerDown={handleTrackPointerDown}>
      <div aria-hidden="true" className="range-pair-rail" />
      <div
        aria-hidden="true"
        className="range-pair-fill"
        style={rangeFillStyle(lowerPercent, upperPercent)}
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
        {sliderTicks.map((tick) => (
          <span
            data-label={tick.label}
            key={`${tick.value}-${tick.label}`}
            style={rangePositionStyle(rangePercent(tick.value, min, max))}
          />
        ))}
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
      style={rangePositionStyle(percent)}
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

function rangePositionStyle(percent: number): CSSProperties {
  return { "--range-pair-ratio": `${percent / 100}` } as CSSProperties;
}

function rangeFillStyle(lowerPercent: number, upperPercent: number): CSSProperties {
  return {
    "--range-pair-min-ratio": `${lowerPercent / 100}`,
    "--range-pair-max-ratio": `${upperPercent / 100}`,
  } as CSSProperties;
}

function snapToStep(value: number, min: number, step: number) {
  const snapped = min + Math.round((value - min) / step) * step;
  const decimals = step.toString().split(".")[1]?.length ?? 0;
  return Number(snapped.toFixed(decimals));
}

function parseEditableNumberInput(rawValue: string) {
  if (rawValue.trim().length === 0) {
    return null;
  }
  const parsed = Number(rawValue);
  return Number.isFinite(parsed) ? parsed : null;
}

function normalizeRangeValue(value: number, min: number, max: number, step: number) {
  return clamp(snapToStep(value, min, step), min, max);
}
