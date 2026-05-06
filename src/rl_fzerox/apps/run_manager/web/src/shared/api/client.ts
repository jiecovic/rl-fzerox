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
  type ManagedRunMetricSample,
  type ManagedTemplate,
  openRunDirectoryResponseSchema,
  type PolicyArchitecturePreview,
  policyArchitecturePreviewSchema,
  resetTrackSamplingResponseSchema,
  runMetricsResponseSchema,
  runsResponseSchema,
  runTrackSamplingResponseSchema,
  type TrackSamplingRuntimeState,
  templatesResponseSchema,
  watchRunResponseSchema,
} from "@/shared/api/contract";

export type RunMetricRangeMode = "recent" | "full";
export const API_SCHEMA_MISMATCH_MESSAGE = "Run-manager backend is outdated. Restart run-manager.";

export class ApiSchemaMismatchError extends Error {
  constructor() {
    super(API_SCHEMA_MISMATCH_MESSAGE);
    this.name = "ApiSchemaMismatchError";
  }
}

const RECENT_RUN_METRIC_LIMIT = 240;
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

export async function fetchRunMetrics(
  runId: string,
  rangeMode: RunMetricRangeMode = "recent",
): Promise<ManagedRunMetricSample[]> {
  return fetchFreshRunMetrics(runId, rangeMode);
}

export async function fetchRunTrackSamplingState(
  runId: string,
): Promise<TrackSamplingRuntimeState | null> {
  const payload = parseApiPayload(
    runTrackSamplingResponseSchema,
    await getJson(`/api/runs/${encodeURIComponent(runId)}/track-sampling`),
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
  const cached = runMetricsCache.get(runMetricsCacheKey(runId, rangeMode));
  if (cached !== undefined) {
    return cached;
  }
  if (rangeMode === "recent") {
    const full = runMetricsCache.get(runMetricsCacheKey(runId, "full"));
    return full === undefined ? null : full.slice(-RECENT_RUN_METRIC_LIMIT);
  }
  return null;
}

export async function fetchFreshRunMetrics(
  runId: string,
  rangeMode: RunMetricRangeMode = "recent",
): Promise<ManagedRunMetricSample[]> {
  return refreshRunMetrics(runId, rangeMode);
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
): Promise<ManagedRun> {
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
): Promise<ManagedRun> {
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

export async function stopRun(runId: string): Promise<ManagedRun> {
  return postRunAction(`/api/runs/${encodeURIComponent(runId)}/stop`);
}

export async function resumeRun(runId: string): Promise<ManagedRun> {
  return postRunAction(`/api/runs/${encodeURIComponent(runId)}/resume`);
}

export async function renameRun(runId: string, name: string): Promise<ManagedRun> {
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

async function getJson(url: string): Promise<unknown> {
  const response = await fetch(url);
  return parseJson(response);
}

async function refreshRunMetrics(
  runId: string,
  rangeMode: RunMetricRangeMode,
): Promise<ManagedRunMetricSample[]> {
  const cacheKey = runMetricsCacheKey(runId, rangeMode);
  const inflight = inflightRunMetrics.get(cacheKey);
  if (inflight !== undefined) {
    return inflight;
  }
  const request = (async () => {
    const query = new URLSearchParams({ mode: rangeMode });
    if (rangeMode === "recent") {
      query.set("limit", String(RECENT_RUN_METRIC_LIMIT));
    }
    const payload = parseApiPayload(
      runMetricsResponseSchema,
      await getJson(`/api/runs/${encodeURIComponent(runId)}/metrics?${query.toString()}`),
    );
    runMetricsCache.set(cacheKey, payload.samples);
    if (rangeMode === "full") {
      runMetricsCache.set(
        runMetricsCacheKey(runId, "recent"),
        payload.samples.slice(-RECENT_RUN_METRIC_LIMIT),
      );
    }
    return payload.samples;
  })();
  inflightRunMetrics.set(cacheKey, request);
  try {
    return await request;
  } finally {
    inflightRunMetrics.delete(cacheKey);
  }
}

function runMetricsCacheKey(runId: string, rangeMode: RunMetricRangeMode) {
  return `${runId}:${rangeMode}`;
}

async function postRunAction(url: string): Promise<ManagedRun> {
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
