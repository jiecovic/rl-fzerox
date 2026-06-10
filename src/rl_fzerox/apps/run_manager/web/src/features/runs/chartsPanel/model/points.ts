// src/rl_fzerox/apps/run_manager/web/src/features/runs/chartsPanel/model/points.ts
import type { RunPlotPoint } from "@/features/runs/charts/RunPlotCard";
import type { ManagedRun, ManagedRunMetricSample } from "@/shared/api/contract";

const metricPointsCache = new WeakMap<
  readonly ManagedRunMetricSample[],
  Map<string, RunPlotPoint[]>
>();
const envStepRatePointsCache = new WeakMap<
  readonly ManagedRunMetricSample[],
  Map<string, RunPlotPoint[]>
>();

export function metricPoints(samples: ManagedRunMetricSample[], key: string) {
  const cached = cachedPoints(metricPointsCache, samples, key);
  if (cached !== undefined) {
    return cached;
  }
  const points = samples
    .map((sample) => {
      const value = metricValueFromSample(sample, key);
      return value === undefined ? null : { step: chartStep(sample), value };
    })
    .filter((point): point is RunPlotPoint => point !== null);
  return withCachedPoints(metricPointsCache, samples, key, points);
}

export function buildEnvStepRatePoints(run: ManagedRun, samples: ManagedRunMetricSample[]) {
  const cacheKey = `${run.id}:${run.started_at ?? run.created_at}`;
  const cached = cachedPoints(envStepRatePointsCache, samples, cacheKey);
  if (cached !== undefined) {
    return cached;
  }
  const points = samples
    .map((sample, index) => {
      const value = sampleEnvStepRateValue(run, samples, index);
      return value === null ? null : { step: chartStep(sample), value };
    })
    .filter((point): point is RunPlotPoint => point !== null);
  return withCachedPoints(envStepRatePointsCache, samples, cacheKey, points);
}

function metricValueFromSample(sample: ManagedRunMetricSample, key: string) {
  const metricValue = sample.metrics[key];
  return metricValue === undefined ? undefined : metricValue;
}

function sampleEnvStepRateValue(run: ManagedRun, samples: ManagedRunMetricSample[], index: number) {
  const sample = samples[index];
  if (sample === undefined) {
    return null;
  }
  const loggedFps = sample.metrics["time/fps"];
  if (loggedFps !== undefined) {
    return loggedFps;
  }
  if (sample.fps !== null && sample.fps !== undefined) {
    return sample.fps;
  }
  const currentMs = Date.parse(sample.created_at);
  if (Number.isNaN(currentMs)) {
    return null;
  }
  const previous = index > 0 ? samples[index - 1] : null;
  if (previous !== null) {
    const previousMs = Date.parse(previous.created_at);
    const elapsedSeconds = (currentMs - previousMs) / 1000;
    if (!Number.isNaN(previousMs) && elapsedSeconds > 0) {
      return Math.max(0, sample.num_timesteps - previous.num_timesteps) / elapsedSeconds;
    }
  }
  const startedAt = run.started_at ?? run.created_at;
  const startedMs = Date.parse(startedAt);
  const elapsedSeconds = (currentMs - startedMs) / 1000;
  if (Number.isNaN(startedMs) || elapsedSeconds <= 0) {
    return null;
  }
  return sample.num_timesteps / elapsedSeconds;
}

function chartStep(sample: ManagedRunMetricSample) {
  return sample.lineage_num_timesteps;
}

function cachedPoints(
  cache: WeakMap<readonly ManagedRunMetricSample[], Map<string, RunPlotPoint[]>>,
  samples: readonly ManagedRunMetricSample[],
  key: string,
) {
  return cache.get(samples)?.get(key);
}

function withCachedPoints(
  cache: WeakMap<readonly ManagedRunMetricSample[], Map<string, RunPlotPoint[]>>,
  samples: readonly ManagedRunMetricSample[],
  key: string,
  points: RunPlotPoint[],
) {
  const adjusted = withForkBoundaryPoint(samples, points);
  const entry = cache.get(samples);
  if (entry === undefined) {
    cache.set(samples, new Map([[key, adjusted]]));
  } else {
    entry.set(key, adjusted);
  }
  return adjusted;
}

function withForkBoundaryPoint(
  samples: readonly ManagedRunMetricSample[],
  points: readonly RunPlotPoint[],
): RunPlotPoint[] {
  const firstSample = samples[0];
  const firstPoint = points[0];
  if (firstSample === undefined || firstPoint === undefined) {
    return [...points];
  }
  const offset = firstSample.lineage_num_timesteps - firstSample.num_timesteps;
  if (offset <= 0 || firstPoint.step <= offset) {
    return [...points];
  }
  return [{ step: offset, value: firstPoint.value }, ...points];
}
