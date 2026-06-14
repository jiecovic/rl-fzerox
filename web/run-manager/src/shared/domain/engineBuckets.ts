// web/run-manager/src/shared/domain/engineBuckets.ts
export function centeredEngineBuckets({
  bucketSize,
  minimum,
  maximum,
}: {
  bucketSize: number;
  minimum: number;
  maximum: number;
}) {
  const lower = clampRawEngineValue(Math.trunc(minimum));
  const upper = clampRawEngineValue(Math.trunc(maximum));
  if (lower > upper) {
    return [];
  }
  const step = Math.max(1, Math.trunc(bucketSize));
  const center = 50;
  const values = new Set<number>();
  for (let offset = 0; ; offset += step) {
    const low = center - offset;
    const high = center + offset;
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

function clampRawEngineValue(value: number) {
  return Math.max(0, Math.min(100, value));
}
