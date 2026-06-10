// src/rl_fzerox/apps/run_manager/web/src/shared/api/client/resources/metadata.ts
import { getJson, parseApiPayload, parseJson, type RequestOptions } from "@/shared/api/client/http";
import {
  type ConfigMetadata,
  configMetadataSchema,
  type ManagedRunConfig,
  type ManagedTemplate,
  type PolicyArchitecturePreview,
  policyArchitecturePreviewSchema,
  templatesResponseSchema,
} from "@/shared/api/contract";

export async function fetchTemplates(): Promise<ManagedTemplate[]> {
  const payload = parseApiPayload(templatesResponseSchema, await getJson("/api/templates"));
  return payload.templates;
}

export async function fetchConfigMetadata(): Promise<ConfigMetadata> {
  return parseApiPayload(configMetadataSchema, await getJson("/api/config-metadata"));
}

export async function fetchPolicyPreview(
  config: ManagedRunConfig,
  options: RequestOptions = {},
): Promise<PolicyArchitecturePreview> {
  const response = await fetch("/api/policy-preview", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(config),
    signal: options.signal,
  });
  return parseApiPayload(policyArchitecturePreviewSchema, await parseJson(response));
}
