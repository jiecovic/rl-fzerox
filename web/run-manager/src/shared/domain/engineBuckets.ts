// web/run-manager/src/shared/domain/engineBuckets.ts
export const ENGINE_SLIDER_STEP_MIN = 0;
export const ENGINE_SLIDER_STEP_MAX = 128;
export const ENGINE_SLIDER_STEP_CENTER = 64;

export function centeredEngineBuckets({
  sliderSpacing,
  minimum,
  maximum,
}: {
  sliderSpacing: number;
  minimum: number;
  maximum: number;
}) {
  const lower = clampRawEngineValue(Math.trunc(minimum));
  const upper = clampRawEngineValue(Math.trunc(maximum));
  if (lower > upper) {
    return [];
  }
  const step = Math.max(1, Math.trunc(sliderSpacing));
  const values = new Set<number>([lower, upper]);
  for (let offset = 0; ; offset += step) {
    const low = ENGINE_SLIDER_STEP_CENTER - offset;
    const high = ENGINE_SLIDER_STEP_CENTER + offset;
    if (lower <= low && low <= upper) {
      values.add(low);
    }
    if (offset !== 0 && lower <= high && high <= upper) {
      values.add(high);
    }
    if (low < lower && high > upper) {
      break;
    }
  }
  return [...values].sort((left, right) => left - right);
}

export function clampRawEngineValue(value: number) {
  return Math.max(ENGINE_SLIDER_STEP_MIN, Math.min(ENGINE_SLIDER_STEP_MAX, value));
}

export function enginePercentToSliderStep(percent: number) {
  return clampRawEngineValue(
    Math.floor((Math.max(0, Math.min(100, percent)) / 100) * ENGINE_SLIDER_STEP_MAX + 0.5),
  );
}

export function engineSliderStepToPercent(step: number) {
  return (clampRawEngineValue(step) / ENGINE_SLIDER_STEP_MAX) * 100;
}

export function engineSliderStepToDisplayPercent(step: number) {
  return engineSliderStepToPercent(step);
}

export function engineSliderStepPercentLabel(step: number) {
  return `${engineSliderStepToDisplayPercent(step).toFixed(1)}%`;
}

export function engineSliderStepLabel(step: number) {
  return `ENG ${engineSliderStepToDisplayPercent(step).toFixed(1)}`;
}
