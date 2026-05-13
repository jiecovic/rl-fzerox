// src/rl_fzerox/apps/run_manager/web/src/shared/api/client.ts
import type { ZodType } from "zod";
import {
  type ConfigMetadata,
  configMetadataSchema,
  createDraftResponseSchema,
  createRunResponseSchema,
  deleteRunResponseSchema,
  draftsResponseSchema,
  forkRunResponseSchema,
  type ManagedDraft,
  type ManagedRun,
  type ManagedRunConfig,
  type ManagedRunDetail,
  type ManagedRunMetricSample,
  type ManagedTemplate,
  openRunDirectoryResponseSchema,
  type PolicyArchitecturePreview,
  policyArchitecturePreviewSchema,
  rebuildTensorboardViewsResponseSchema,
  resetTrackSamplingResponseSchema,
  runMetricsResponseSchema,
  runResponseSchema,
  runsResponseSchema,
  runTrackSamplingResponseSchema,
  type TensorboardViewGroup,
  type TrackSamplingRuntimeState,
  templatesResponseSchema,
  updateLineageGroupsResponseSchema,
  watchRunResponseSchema,
} from "@/shared/api/contract";

export type RunMetricRangeMode = "recent" | "full";
export const API_SCHEMA_MISMATCH_MESSAGE = "Run-manager backend is outdated. Restart run-manager.";

interface RequestOptions {
  signal?: AbortSignal;
}

export class ApiSchemaMismatchError extends Error {
  constructor() {
    super(API_SCHEMA_MISMATCH_MESSAGE);
    this.name = "ApiSchemaMismatchError";
  }
}

const RECENT_RUN_METRIC_LIMIT = 240;
const MAX_RECENT_RUN_METRIC_CACHE_ENTRIES = 48;
const MAX_FULL_RUN_METRIC_CACHE_ENTRIES = 12;
const runMetricsCache = new Map<string, ManagedRunMetricSample[]>();
const inflightRunMetrics = new Map<string, Promise<ManagedRunMetricSample[]>>();

export async function fetchTemplates(): Promise<ManagedTemplate[]> {
  const payload = parseApiPayload(templatesResponseSchema, await getJson("/api/templates"));
  return payload.templates;
}

export async function fetchDrafts(): Promise<ManagedDraft[]> {
  const payload = parseApiPayload(draftsResponseSchema, await getJson("/api/drafts"));
  return payload.drafts;
}

export async function fetchRuns(): Promise<ManagedRun[]> {
  const payload = parseApiPayload(runsResponseSchema, await getJson("/api/runs"));
  return payload.runs;
}

export async function fetchRun(runId: string): Promise<ManagedRunDetail> {
  const payload = parseApiPayload(
    runResponseSchema,
    await getJson(`/api/runs/${encodeURIComponent(runId)}`),
  );
  return payload.run;
}

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

export async function fetchConfigMetadata(): Promise<ConfigMetadata> {
  return parseApiPayload(configMetadataSchema, await getJson("/api/config-metadata"));
}

export async function fetchPolicyPreview(
  config: ManagedRunConfig,
): Promise<PolicyArchitecturePreview> {
  const response = await postPolicyPreview(config);
  return parseApiPayload(policyArchitecturePreviewSchema, await parseJson(response));
}

async function postPolicyPreview(config: ManagedRunConfig): Promise<Response> {
  return fetch("/api/policy-preview", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(config),
  });
}

export async function createDraft(name: string, config: ManagedRunConfig): Promise<ManagedDraft> {
  return createDraftWithSource(name, config, null, null);
}

export async function createDraftWithSource(
  name: string,
  config: ManagedRunConfig,
  sourceRunId: string | null,
  sourceArtifact: "latest" | "best" | null,
): Promise<ManagedDraft> {
  const response = await fetch("/api/drafts", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      name,
      config,
      source_run_id: sourceRunId,
      source_artifact: sourceArtifact,
    }),
  });
  const payload = parseApiPayload(createDraftResponseSchema, await parseJson(response));
  return payload.draft;
}

export async function updateDraft(
  id: string,
  name: string,
  config: ManagedRunConfig,
): Promise<ManagedDraft> {
  return updateDraftWithSource(id, name, config, null, null);
}

export async function updateDraftWithSource(
  id: string,
  name: string,
  config: ManagedRunConfig,
  sourceRunId: string | null,
  sourceArtifact: "latest" | "best" | null,
): Promise<ManagedDraft> {
  const response = await fetch(`/api/drafts/${encodeURIComponent(id)}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      name,
      config,
      source_run_id: sourceRunId,
      source_artifact: sourceArtifact,
    }),
  });
  const payload = parseApiPayload(createDraftResponseSchema, await parseJson(response));
  return payload.draft;
}

export async function launchRun(
  name: string,
  config: ManagedRunConfig,
  draftId: string | null,
  sourceRunId: string | null = null,
  sourceArtifact: "latest" | "best" | null = null,
): Promise<ManagedRunDetail> {
  const response = await fetch("/api/runs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      name,
      config,
      draft_id: draftId,
      source_run_id: sourceRunId,
      source_artifact: sourceArtifact,
    }),
  });
  const payload = parseApiPayload(createRunResponseSchema, await parseJson(response));
  return payload.run;
}

export async function forkRun(
  runId: string,
  artifact: "latest" | "best",
  name?: string,
  config?: ManagedRunConfig,
): Promise<ManagedRunDetail> {
  const response = await fetch(`/api/runs/${encodeURIComponent(runId)}/fork`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      artifact,
      ...(name === undefined ? {} : { name }),
      ...(config === undefined ? {} : { config }),
    }),
  });
  const payload = parseApiPayload(forkRunResponseSchema, await parseJson(response));
  return payload.run;
}

export async function stopRun(runId: string): Promise<ManagedRunDetail> {
  return postRunAction(`/api/runs/${encodeURIComponent(runId)}/stop`);
}

export async function resumeRun(runId: string): Promise<ManagedRunDetail> {
  return postRunAction(`/api/runs/${encodeURIComponent(runId)}/resume`);
}

export async function renameRun(runId: string, name: string): Promise<ManagedRunDetail> {
  const response = await fetch(`/api/runs/${encodeURIComponent(runId)}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
  const payload = parseApiPayload(createRunResponseSchema, await parseJson(response));
  return payload.run;
}

export async function openRunDirectory(runId: string): Promise<void> {
  const response = await fetch(`/api/runs/${encodeURIComponent(runId)}/open-dir`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
  });
  parseApiPayload(openRunDirectoryResponseSchema, await parseJson(response));
}

export async function watchRun(
  runId: string,
  artifact: "latest" | "best",
): Promise<"started" | "already_running"> {
  const response = await fetch(
    `/api/runs/${encodeURIComponent(runId)}/watch?artifact=${encodeURIComponent(artifact)}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
    },
  );
  const payload = parseApiPayload(watchRunResponseSchema, await parseJson(response));
  return payload.status;
}

export async function deleteDraft(id: string): Promise<void> {
  const response = await fetch(`/api/drafts/${encodeURIComponent(id)}`, { method: "DELETE" });
  await parseJson(response);
}

export async function deleteRun(id: string): Promise<void> {
  const response = await fetch(`/api/runs/${encodeURIComponent(id)}`, { method: "DELETE" });
  parseApiPayload(deleteRunResponseSchema, await parseJson(response));
}

export async function deleteLineage(id: string): Promise<void> {
  const response = await fetch(`/api/lineages/${encodeURIComponent(id)}`, { method: "DELETE" });
  parseApiPayload(deleteRunResponseSchema, await parseJson(response));
}

export async function updateLineageGroups(
  lineageId: string,
  groupNames: readonly string[],
): Promise<string[]> {
  const response = await fetch(`/api/lineages/${encodeURIComponent(lineageId)}/groups`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ group_names: groupNames }),
  });
  const payload = parseApiPayload(updateLineageGroupsResponseSchema, await parseJson(response));
  return payload.lineage_groups;
}

export async function rebuildTensorboardViews(): Promise<TensorboardViewGroup[]> {
  const response = await fetch("/api/tensorboard-views/rebuild", { method: "POST" });
  const payload = parseApiPayload(rebuildTensorboardViewsResponseSchema, await parseJson(response));
  return payload.tensorboard_views;
}

async function getJson(url: string, options: RequestOptions = {}): Promise<unknown> {
  const response = await fetch(url, { signal: options.signal });
  return parseJson(response);
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
    if (rangeMode === "full") {
      setRunMetricCacheEntry(
        runMetricsCacheKey(runId, "recent"),
        payload.samples.slice(-RECENT_RUN_METRIC_LIMIT),
      );
    }
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

async function postRunAction(url: string): Promise<ManagedRunDetail> {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
  });
  const payload = parseApiPayload(createRunResponseSchema, await parseJson(response));
  return payload.run;
}

function parseApiPayload<T>(schema: ZodType<T>, payload: unknown): T {
  const parsed = schema.safeParse(payload);
  if (!parsed.success) {
    throw new ApiSchemaMismatchError();
  }
  return parsed.data;
}

async function parseJson(response: Response): Promise<unknown> {
  const payload = (await response.json()) as { error?: unknown };
  if (!response.ok) {
    throw new Error(typeof payload.error === "string" ? payload.error : response.statusText);
  }
  return payload;
}
