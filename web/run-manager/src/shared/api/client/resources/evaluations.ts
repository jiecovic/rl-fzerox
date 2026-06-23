// web/run-manager/src/shared/api/client/resources/evaluations.ts
import { getJson, parseApiPayload, parseJson } from "@/shared/api/client/http";
import {
  type CreateEvaluationRequest,
  createEvaluationResponseSchema,
  type EvaluationsResponse,
  evaluationsResponseSchema,
  type ManagedEvaluation,
  startEvaluationResponseSchema,
  type UpdateEvaluationRequest,
  updateEvaluationResponseSchema,
} from "@/shared/api/contract";

export async function fetchEvaluationData(): Promise<EvaluationsResponse> {
  return parseApiPayload(evaluationsResponseSchema, await getJson("/api/evaluations"));
}

export async function fetchEvaluations(): Promise<ManagedEvaluation[]> {
  return (await fetchEvaluationData()).evaluations;
}

export async function createEvaluation(
  request: CreateEvaluationRequest,
): Promise<ManagedEvaluation> {
  const response = await fetch("/api/evaluations", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      name: request.name,
      source_run_id: request.sourceRunId,
      preset_id: request.presetId,
      source_artifact: request.sourceArtifact,
      policy_mode: request.policyMode,
      seed: request.seed,
      config: request.config,
      target: {
        mode: request.targetMode,
        course_ids: request.courseIds,
        cup_ids: request.cupIds,
        difficulties: request.difficulties,
        vehicle_ids: request.vehicleIds,
        repeats_per_target: request.repeatsPerTarget,
      },
    }),
  });
  const payload = parseApiPayload(createEvaluationResponseSchema, await parseJson(response));
  return payload.evaluation;
}

export async function deleteEvaluation(evaluationId: string): Promise<boolean> {
  const response = await fetch(`/api/evaluations/${encodeURIComponent(evaluationId)}`, {
    method: "DELETE",
  });
  const payload = await parseJson(response);
  if (
    typeof payload === "object" &&
    payload !== null &&
    "deleted" in payload &&
    typeof payload.deleted === "boolean"
  ) {
    return payload.deleted;
  }
  throw new Error("run-manager delete evaluation response is invalid");
}

export async function startEvaluation(evaluationId: string): Promise<ManagedEvaluation> {
  const response = await fetch(`/api/evaluations/${encodeURIComponent(evaluationId)}/start`, {
    method: "POST",
  });
  const payload = parseApiPayload(startEvaluationResponseSchema, await parseJson(response));
  return payload.evaluation;
}

export async function updateEvaluation(
  evaluationId: string,
  request: UpdateEvaluationRequest,
): Promise<ManagedEvaluation> {
  const response = await fetch(`/api/evaluations/${encodeURIComponent(evaluationId)}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name: request.name }),
  });
  const payload = parseApiPayload(updateEvaluationResponseSchema, await parseJson(response));
  return payload.evaluation;
}
