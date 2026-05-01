import {
  type ConfigMetadata,
  configMetadataSchema,
  createDraftResponseSchema,
  draftsResponseSchema,
  type ManagedDraft,
  type ManagedRun,
  type ManagedRunConfig,
  type ManagedTemplate,
  type PolicyArchitecturePreview,
  policyArchitecturePreviewSchema,
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

export async function fetchConfigMetadata(): Promise<ConfigMetadata> {
  return configMetadataSchema.parse(await getJson("/api/config-metadata"));
}

export async function fetchPolicyPreview(
  config: ManagedRunConfig,
): Promise<PolicyArchitecturePreview> {
  let response = await postPolicyPreview(config);
  if (!response.ok) {
    response = await postPolicyPreview(legacyPreviewConfig(config));
  }
  return policyArchitecturePreviewSchema.parse(await parseJson(response));
}

async function postPolicyPreview(config: unknown): Promise<Response> {
  return fetch("/api/policy-preview", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(config),
  });
}

function legacyPreviewConfig(config: ManagedRunConfig): unknown {
  const policy = { ...config.policy };
  delete (policy as Partial<typeof policy>).activation;
  return { ...config, policy };
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

export async function updateDraft(
  id: string,
  name: string,
  config: ManagedRunConfig,
): Promise<ManagedDraft> {
  const response = await fetch(`/api/drafts/${encodeURIComponent(id)}`, {
    method: "PUT",
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
