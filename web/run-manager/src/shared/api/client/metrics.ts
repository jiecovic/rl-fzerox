// web/run-manager/src/shared/api/client/metrics.ts

import { getJson, parseApiPayload, parseJson, type RequestOptions } from "@/shared/api/client/http";
import {
  type EngineTuningRuntimeState,
  type ManagedRunMetricSample,
  resetTrackSamplingResponseSchema,
  runEngineTuningResponseSchema,
  runMetricsResponseSchema,
  runTrackSamplingResponseSchema,
  type TrackSamplingRuntimeState,
} from "@/shared/api/contract";

export type RunMetricRangeMode = "recent" | "full";

const RECENT_RUN_METRIC_LIMIT = 240;
const MAX_RECENT_RUN_METRIC_CACHE_ENTRIES = 48;
const MAX_FULL_RUN_METRIC_CACHE_ENTRIES = 12;
const runMetricsCache = new Map<string, ManagedRunMetricSample[]>();
const inflightRunMetrics = new Map<string, Promise<ManagedRunMetricSample[]>>();

export async function fetchRunMetrics(
  runId: string,
  rangeMode: RunMetricRangeMode = "recent",
  options: RequestOptions = {},
): Promise<ManagedRunMetricSample[]> {
  return fetchFreshRunMetrics(runId, rangeMode, options);
}

export async function fetchRunTrackSamplingState(
  runId: string,
  options: RequestOptions = {},
): Promise<TrackSamplingRuntimeState | null> {
  const payload = parseApiPayload(
    runTrackSamplingResponseSchema,
    await getJson(`/api/runs/${encodeURIComponent(runId)}/track-sampling`, options),
  );
  return payload.state;
}

export async function fetchRunEngineTuningState(
  runId: string,
  artifact: "latest" | "best" | "final" = "latest",
  options: RequestOptions = {},
): Promise<{ enabled: boolean; state: EngineTuningRuntimeState | null }> {
  const query = new URLSearchParams({ artifact });
  return parseApiPayload(
    runEngineTuningResponseSchema,
    await getJson(
      `/api/runs/${encodeURIComponent(runId)}/engine-tuning?${query.toString()}`,
      options,
    ),
  );
}

export async function resetRunTrackSamplingState(runId: string): Promise<void> {
  const response = await fetch(`/api/runs/${encodeURIComponent(runId)}/track-sampling/reset`, {
    method: "POST",
  });
  parseApiPayload(resetTrackSamplingResponseSchema, await parseJson(response));
}

export function getCachedRunMetrics(
  runId: string,
  rangeMode: RunMetricRangeMode = "recent",
): ManagedRunMetricSample[] | null {
  const cacheKey = runMetricsCacheKey(runId, rangeMode);
  const cached = getRunMetricCacheEntry(cacheKey);
  if (cached !== undefined) {
    return cached;
  }
  if (rangeMode === "recent") {
    const full = getRunMetricCacheEntry(runMetricsCacheKey(runId, "full"));
    return full === undefined ? null : full.slice(-RECENT_RUN_METRIC_LIMIT);
  }
  return null;
}

export async function fetchFreshRunMetrics(
  runId: string,
  rangeMode: RunMetricRangeMode = "recent",
  options: RequestOptions = {},
): Promise<ManagedRunMetricSample[]> {
  return refreshRunMetrics(runId, rangeMode, options);
}

async function refreshRunMetrics(
  runId: string,
  rangeMode: RunMetricRangeMode,
  options: RequestOptions,
): Promise<ManagedRunMetricSample[]> {
  const cacheKey = runMetricsCacheKey(runId, rangeMode);
  if (options.signal === undefined) {
    const inflight = inflightRunMetrics.get(cacheKey);
    if (inflight !== undefined) {
      return inflight;
    }
  }
  const request = (async () => {
    const query = new URLSearchParams({ mode: rangeMode });
    if (rangeMode === "recent") {
      query.set("limit", String(RECENT_RUN_METRIC_LIMIT));
    }
    const payload = parseApiPayload(
      runMetricsResponseSchema,
      await getJson(`/api/runs/${encodeURIComponent(runId)}/metrics?${query.toString()}`, options),
    );
    setRunMetricCacheEntry(cacheKey, payload.samples);
    return payload.samples;
  })();
  if (options.signal === undefined) {
    inflightRunMetrics.set(cacheKey, request);
  }
  try {
    return await request;
  } finally {
    if (options.signal === undefined) {
      inflightRunMetrics.delete(cacheKey);
    }
  }
}

function runMetricsCacheKey(runId: string, rangeMode: RunMetricRangeMode) {
  return `${runId}:${rangeMode}`;
}

function getRunMetricCacheEntry(cacheKey: string): ManagedRunMetricSample[] | undefined {
  const cached = runMetricsCache.get(cacheKey);
  if (cached === undefined) {
    return undefined;
  }
  runMetricsCache.delete(cacheKey);
  runMetricsCache.set(cacheKey, cached);
  return cached;
}

function setRunMetricCacheEntry(cacheKey: string, samples: ManagedRunMetricSample[]) {
  runMetricsCache.delete(cacheKey);
  runMetricsCache.set(cacheKey, samples);
  trimRunMetricCache("recent", MAX_RECENT_RUN_METRIC_CACHE_ENTRIES);
  trimRunMetricCache("full", MAX_FULL_RUN_METRIC_CACHE_ENTRIES);
}

function trimRunMetricCache(rangeMode: RunMetricRangeMode, maxEntries: number) {
  const keys = [...runMetricsCache.keys()].filter((key) => key.endsWith(`:${rangeMode}`));
  for (const key of keys.slice(0, Math.max(0, keys.length - maxEntries))) {
    runMetricsCache.delete(key);
  }
}
