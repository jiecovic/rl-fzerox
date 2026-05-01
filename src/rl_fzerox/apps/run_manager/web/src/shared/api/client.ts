import {
  createDraftResponseSchema,
  draftsResponseSchema,
  type ManagedDraft,
  type ManagedRun,
  type ManagedRunConfig,
  type ManagedTemplate,
  runsResponseSchema,
  templatesResponseSchema,
} from "@/shared/api/contract";

export async function fetchTemplates(): Promise<ManagedTemplate[]> {
  const payload = templatesResponseSchema.parse(await getJson("/api/templates"));
  return payload.templates;
}

export async function fetchDrafts(): Promise<ManagedDraft[]> {
  const payload = draftsResponseSchema.parse(await getJson("/api/drafts"));
  return payload.drafts;
}

export async function fetchRuns(): Promise<ManagedRun[]> {
  const payload = runsResponseSchema.parse(await getJson("/api/runs"));
  return payload.runs;
}

export async function createDraft(name: string, config: ManagedRunConfig): Promise<ManagedDraft> {
  const response = await fetch("/api/drafts", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, config }),
  });
  const payload = createDraftResponseSchema.parse(await parseJson(response));
  return payload.draft;
}

export async function deleteDraft(id: string): Promise<void> {
  const response = await fetch(`/api/drafts/${encodeURIComponent(id)}`, { method: "DELETE" });
  await parseJson(response);
}

async function getJson(url: string): Promise<unknown> {
  const response = await fetch(url);
  return parseJson(response);
}

async function parseJson(response: Response): Promise<unknown> {
  const payload = (await response.json()) as { error?: unknown };
  if (!response.ok) {
    throw new Error(typeof payload.error === "string" ? payload.error : response.statusText);
  }
  return payload;
}
