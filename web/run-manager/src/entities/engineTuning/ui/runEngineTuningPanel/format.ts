// web/run-manager/src/entities/engineTuning/ui/runEngineTuningPanel/format.ts
import { engineSliderStepLabel } from "@/shared/domain/engineBuckets";
import { formatRaceTimeMs } from "@/shared/ui/format";

export function engineStepLabel(step: number) {
  return engineSliderStepLabel(step);
}

export function formatOptionalRaceTime(value: number | null) {
  return value === null ? "no time yet" : formatRaceTime(value);
}

export function formatOptionalScore(value: number | null) {
  return value === null ? "no score yet" : formatScore(value);
}

export function formatPercent(value: number | null) {
  if (value === null) {
    return "-";
  }
  const percent = value * 100;
  return `${percent.toFixed(Math.abs(percent) < 1 ? 2 : 1)}%`;
}

export function formatRaceTime(value: number) {
  return formatRaceTimeMs(value);
}

export function formatScore(value: number) {
  return value.toLocaleString(undefined, {
    maximumFractionDigits: Math.abs(value) < 10 ? 2 : 1,
    minimumFractionDigits: 0,
  });
}

export function humanizeKey(key: string) {
  return key
    .split("_")
    .filter((part) => part.length > 0)
    .map((part) => part[0]?.toUpperCase() + part.slice(1))
    .join(" ");
}
