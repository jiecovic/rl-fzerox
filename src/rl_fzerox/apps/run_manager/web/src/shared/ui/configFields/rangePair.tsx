// src/rl_fzerox/apps/run_manager/web/src/shared/ui/configFields/rangePair.tsx
import type { CSSProperties } from "react";
import { useCallback, useEffect, useRef, useState } from "react";
import { cn } from "@/shared/ui/cn";
import { clamp, formatCompactDecimal, formatCompactNumber } from "@/shared/ui/configFields/format";
import { FieldLabel } from "@/shared/ui/configFields/label";
import {
  blurOnEnter,
  editableNumberInputProps,
  parseDecimalInput,
  useEditableNumberInput,
} from "@/shared/ui/configFields/numberInput";
import type { SliderTick } from "@/shared/ui/configFields/types";
import { FieldInput, FieldShell, SwitchButton } from "@/shared/ui/Field";

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
  const fieldResetHandler =
    resetMin === undefined ||
    resetMax === undefined ||
    (Object.is(valueMin, resetMin) && Object.is(valueMax, resetMax))
      ? undefined
      : () => onChange({ min: resetMin, max: resetMax });
  const minInput = useEditableNumberInput({
    format: String,
    formattedValue: String(valueMin),
    normalize: (nextValue) => normalizeRangeValue(nextValue, min, valueMax, step),
    onCommit: (normalized) => onChange({ min: normalized, max: valueMax }),
    parse: parseDecimalInput,
  });
  const maxInput = useEditableNumberInput({
    format: String,
    formattedValue: String(valueMax),
    normalize: (nextValue) => normalizeRangeValue(nextValue, valueMin, max, step),
    onCommit: (normalized) => onChange({ min: valueMin, max: normalized }),
    parse: parseDecimalInput,
  });

  function updateMin(nextValue: number) {
    const normalized = normalizeRangeValue(nextValue, min, valueMax, step);
    onChange({ min: normalized, max: valueMax });
    minInput.setRawValue(String(normalized));
  }

  function updateMax(nextValue: number) {
    const normalized = normalizeRangeValue(nextValue, valueMin, max, step);
    onChange({ min: valueMin, max: normalized });
    maxInput.setRawValue(String(normalized));
  }

  return (
    <FieldShell className="range-pair-field">
      <FieldLabel help={help} label={label} onReset={fieldResetHandler} />
      <div className="grid grid-cols-[minmax(0,1fr)_148px] items-center gap-3">
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
        <div className="grid grid-cols-2 gap-2">
          <FieldInput
            aria-label={`${label} minimum`}
            className="h-[34px] indent-0 tabular-nums"
            max={valueMax}
            min={min}
            step={step}
            {...editableNumberInputProps("decimal")}
            value={minInput.rawValue}
            onBlur={minInput.commitRawValue}
            onChange={(event) => minInput.changeRawValue(event.target.value)}
            onKeyDown={blurOnEnter}
          />
          <FieldInput
            aria-label={`${label} maximum`}
            className="h-[34px] indent-0 tabular-nums"
            max={max}
            min={valueMin}
            step={step}
            {...editableNumberInputProps("decimal")}
            value={maxInput.rawValue}
            onBlur={maxInput.commitRawValue}
            onChange={(event) => maxInput.changeRawValue(event.target.value)}
            onKeyDown={blurOnEnter}
          />
        </div>
      </div>
    </FieldShell>
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
  const fieldResetHandler =
    resetMin === undefined ||
    resetMax === undefined ||
    (Object.is(valueMin, resetMin) && Object.is(valueMax, resetMax))
      ? undefined
      : () => onChange({ min: resetMin, max: resetMax });
  const minInput = useEditableNumberInput({
    format: String,
    formattedValue: String(activeMin),
    normalize: (nextValue) => normalizeRangeValue(nextValue, min, activeMax, step),
    onCommit: (normalized) => {
      if (enabled) {
        onChange({ min: normalized, max: activeMax });
      }
    },
    parse: parseDecimalInput,
  });
  const maxInput = useEditableNumberInput({
    format: String,
    formattedValue: String(activeMax),
    normalize: (nextValue) => normalizeRangeValue(nextValue, activeMin, max, step),
    onCommit: (normalized) => {
      if (enabled) {
        onChange({ min: activeMin, max: normalized });
      }
    },
    parse: parseDecimalInput,
  });

  function updateEnabled() {
    if (enabled) {
      onChange({ min: null, max: null });
      return;
    }
    const normalizedMin = normalizeRangeValue(defaultMin, min, defaultMax, step);
    const normalizedMax = normalizeRangeValue(defaultMax, normalizedMin, max, step);
    onChange({ min: normalizedMin, max: normalizedMax });
    minInput.setRawValue(String(normalizedMin));
    maxInput.setRawValue(String(normalizedMax));
  }

  function updateMin(nextValue: number) {
    if (enabled) {
      const normalized = normalizeRangeValue(nextValue, min, activeMax, step);
      onChange({ min: normalized, max: activeMax });
      minInput.setRawValue(String(normalized));
    }
  }

  function updateMax(nextValue: number) {
    if (enabled) {
      const normalized = normalizeRangeValue(nextValue, activeMin, max, step);
      onChange({ min: activeMin, max: normalized });
      maxInput.setRawValue(String(normalized));
    }
  }

  return (
    <FieldShell className="range-pair-field">
      <FieldLabel help={help} label={label} onReset={fieldResetHandler} />
      <div
        className={cn(
          "grid grid-cols-[52px_minmax(0,1fr)_148px] items-center gap-3",
          enabled
            ? undefined
            : "[&_.range-pair-slider]:pointer-events-none [&_input]:pointer-events-none",
        )}
      >
        <SwitchButton active={enabled} label={`${label} enabled`} onClick={updateEnabled} />
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
        <div className="grid grid-cols-2 gap-2">
          <FieldInput
            aria-label={`${label} minimum`}
            className="h-[34px] indent-0 tabular-nums"
            disabled={!enabled}
            max={activeMax}
            min={min}
            step={step}
            {...editableNumberInputProps("decimal")}
            value={minInput.rawValue}
            onBlur={minInput.commitRawValue}
            onChange={(event) => minInput.changeRawValue(event.target.value)}
            onKeyDown={blurOnEnter}
          />
          <FieldInput
            aria-label={`${label} maximum`}
            className="h-[34px] indent-0 tabular-nums"
            disabled={!enabled}
            max={max}
            min={activeMin}
            step={step}
            {...editableNumberInputProps("decimal")}
            value={maxInput.rawValue}
            onBlur={maxInput.commitRawValue}
            onChange={(event) => maxInput.changeRawValue(event.target.value)}
            onKeyDown={blurOnEnter}
          />
        </div>
      </div>
    </FieldShell>
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

function normalizeRangeValue(value: number, min: number, max: number, step: number) {
  return clamp(snapToStep(value, min, step), min, max);
}
