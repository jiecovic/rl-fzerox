// web/run-manager/src/shared/api/client/resources/runs.ts
import { getJson, parseApiPayload, parseJson } from "@/shared/api/client/http";
import {
  createRunResponseSchema,
  deleteRunResponseSchema,
  forkRunResponseSchema,
  type ManagedRun,
  type ManagedRunConfig,
  type ManagedRunDetail,
  openRunDirectoryResponseSchema,
  type PolicyPlaybackMode,
  runResponseSchema,
  runsResponseSchema,
  updateLineageGroupsResponseSchema,
  type WatchRenderer,
  watchRunResponseSchema,
} from "@/shared/api/contract";

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

export async function launchRun(
  name: string,
  config: ManagedRunConfig,
  draftId: string | null,
  sourceRunId: string | null = null,
  sourceArtifact: "latest" | "best" | null = null,
  copyAltBaselines = true,
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
      copy_alt_baselines: copyAltBaselines,
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
  copyAltBaselines = true,
): Promise<ManagedRunDetail> {
  const response = await fetch(`/api/runs/${encodeURIComponent(runId)}/fork`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      artifact,
      ...(name === undefined ? {} : { name }),
      ...(config === undefined ? {} : { config }),
      copy_alt_baselines: copyAltBaselines,
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
  policyMode: PolicyPlaybackMode,
): Promise<"started" | "already_running"> {
  const response = await fetch(
    `/api/runs/${encodeURIComponent(runId)}/watch?artifact=${encodeURIComponent(artifact)}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ device, renderer, policy_mode: policyMode }),
    },
  );
  const payload = parseApiPayload(watchRunResponseSchema, await parseJson(response));
  return payload.status;
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

async function postRunAction(url: string): Promise<ManagedRunDetail> {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
  });
  const payload = parseApiPayload(createRunResponseSchema, await parseJson(response));
  return payload.run;
}
