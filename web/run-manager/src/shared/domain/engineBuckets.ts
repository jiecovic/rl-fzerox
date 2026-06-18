// web/run-manager/src/shared/domain/engineBuckets.ts
export const ENGINE_SLIDER = {
  minStep: 0,
  maxStep: 128,
  centerStep: 64,
} as const;

export function centeredEngineBuckets({
  sideCount,
  minimum,
  maximum,
}: {
  sideCount: number;
  minimum: number;
  maximum: number;
}) {
  const lower = clampRawEngineValue(Math.trunc(minimum));
  const upper = clampRawEngineValue(Math.trunc(maximum));
  if (lower > upper) {
    return [];
  }
  const count = Math.max(0, Math.trunc(sideCount));
  if (ENGINE_SLIDER.centerStep < lower || ENGINE_SLIDER.centerStep > upper) {
    return [];
  }
  if (count === 0) {
    return [ENGINE_SLIDER.centerStep];
  }
  const values = [
    ...roundedStepSpan(lower, ENGINE_SLIDER.centerStep, count).slice(0, -1),
    ...roundedStepSpan(ENGINE_SLIDER.centerStep, upper, count),
  ];
  const uniqueValues = [...new Set(values)].sort((left, right) => left - right);
  return uniqueValues.length === count * 2 + 1 ? uniqueValues : [];
}

export function bucketSideCountFromRawValues(bucketRawValues: readonly number[]) {
  const normalized = [...new Set(bucketRawValues.map((value) => Math.trunc(value)))].sort(
    (left, right) => left - right,
  );
  const centerIndex = normalized.indexOf(ENGINE_SLIDER.centerStep);
  if (centerIndex < 0) {
    return 0;
  }
  return Math.min(centerIndex, normalized.length - centerIndex - 1);
}

export function clampRawEngineValue(value: number) {
  return Math.max(ENGINE_SLIDER.minStep, Math.min(ENGINE_SLIDER.maxStep, value));
}

export function enginePercentToSliderStep(percent: number) {
  return clampRawEngineValue(
    Math.floor((Math.max(0, Math.min(100, percent)) / 100) * ENGINE_SLIDER.maxStep + 0.5),
  );
}

export function engineSliderStepToPercent(step: number) {
  return (clampRawEngineValue(step) / ENGINE_SLIDER.maxStep) * 100;
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

function roundedStepSpan(start: number, end: number, intervalCount: number) {
  return Array.from({ length: intervalCount + 1 }, (_, index) =>
    Math.floor(start + ((end - start) * index) / intervalCount + 0.5),
  );
}
