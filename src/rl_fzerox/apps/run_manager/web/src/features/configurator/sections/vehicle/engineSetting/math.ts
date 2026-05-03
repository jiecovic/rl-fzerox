import { clamp } from "@/features/configurator/fields/format";

export function sliderRatio(value: number, min: number, max: number) {
  return clamp((value - min) / (max - min), 0, 1);
}

export function valueFromClientX(
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

function snapToStep(value: number, min: number, step: number) {
  const snapped = min + Math.round((value - min) / step) * step;
  const decimals = step.toString().split(".")[1]?.length ?? 0;
  return Number(snapped.toFixed(decimals));
}

export function parseBoundedInt(rawValue: string, min: number, max: number) {
  const parsed = Number(rawValue.replace(/[,_\s]/g, ""));
  if (!Number.isFinite(parsed)) {
    return min;
  }
  return Math.round(clamp(parsed, min, max));
}
