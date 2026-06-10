// src/rl_fzerox/apps/run_manager/web/src/shared/api/client/resources/drafts.ts
import { parseApiPayload, parseJson } from "@/shared/api/client/http";
import {
  createDraftResponseSchema,
  draftsResponseSchema,
  type ManagedDraft,
  type ManagedRunConfig,
} from "@/shared/api/contract";

export async function fetchDrafts(): Promise<ManagedDraft[]> {
  const payload = parseApiPayload(draftsResponseSchema, await fetchJson("/api/drafts"));
  return payload.drafts;
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

export async function deleteDraft(id: string): Promise<void> {
  const response = await fetch(`/api/drafts/${encodeURIComponent(id)}`, { method: "DELETE" });
  await parseJson(response);
}

async function fetchJson(url: string): Promise<unknown> {
  const response = await fetch(url);
  return parseJson(response);
}
