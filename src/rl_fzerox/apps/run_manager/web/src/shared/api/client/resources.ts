// src/rl_fzerox/apps/run_manager/web/src/shared/api/client/resources.ts

import { getJson, parseApiPayload, parseJson, type RequestOptions } from "@/shared/api/client/http";
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
  type ManagedTemplate,
  openRunDirectoryResponseSchema,
  type PolicyArchitecturePreview,
  policyArchitecturePreviewSchema,
  rebuildTensorboardViewsResponseSchema,
  runResponseSchema,
  runsResponseSchema,
  type TensorboardViewGroup,
  templatesResponseSchema,
  updateLineageGroupsResponseSchema,
  type WatchRenderer,
  watchRunResponseSchema,
} from "@/shared/api/contract";

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

export async function fetchConfigMetadata(): Promise<ConfigMetadata> {
  return parseApiPayload(configMetadataSchema, await getJson("/api/config-metadata"));
}

export async function fetchPolicyPreview(
  config: ManagedRunConfig,
  options: RequestOptions = {},
): Promise<PolicyArchitecturePreview> {
  const response = await postPolicyPreview(config, options);
  return parseApiPayload(policyArchitecturePreviewSchema, await parseJson(response));
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
  device: "cpu" | "cuda",
  renderer: WatchRenderer,
): Promise<"started" | "already_running"> {
  const response = await fetch(
    `/api/runs/${encodeURIComponent(runId)}/watch?artifact=${encodeURIComponent(artifact)}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ device, renderer }),
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

async function postPolicyPreview(
  config: ManagedRunConfig,
  options: RequestOptions,
): Promise<Response> {
  return fetch("/api/policy-preview", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(config),
    signal: options.signal,
  });
}

async function postRunAction(url: string): Promise<ManagedRunDetail> {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
  });
  const payload = parseApiPayload(createRunResponseSchema, await parseJson(response));
  return payload.run;
}
