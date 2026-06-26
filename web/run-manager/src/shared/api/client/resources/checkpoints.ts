// web/run-manager/src/shared/api/client/resources/checkpoints.ts
import { getJson, parseApiPayload, parseJson } from "@/shared/api/client/http";
import {
  type CheckpointCatalogResponse,
  checkpointCatalogResponseSchema,
  type InstallCheckpointResponse,
  installCheckpointResponseSchema,
} from "@/shared/api/contract";

export async function fetchCheckpointCatalog(): Promise<CheckpointCatalogResponse> {
  return parseApiPayload(
    checkpointCatalogResponseSchema,
    await getJson("/api/checkpoints/catalog"),
  );
}

export async function installCatalogCheckpoint(
  checkpointId: string,
  version: string,
): Promise<InstallCheckpointResponse> {
  const response = await fetch(
    `/api/checkpoints/catalog/${encodeURIComponent(checkpointId)}/${encodeURIComponent(
      version,
    )}/install`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
    },
  );
  return parseApiPayload(installCheckpointResponseSchema, await parseJson(response));
}
