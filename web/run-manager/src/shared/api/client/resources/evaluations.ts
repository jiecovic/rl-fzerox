// web/run-manager/src/shared/api/client/resources/evaluations.ts
import { getJson, parseApiPayload, parseJson } from "@/shared/api/client/http";
import {
  type CreateEvaluationPresetRequest,
  type CreateEvaluationRequest,
  cancelEvaluationResponseSchema,
  createEvaluationPresetResponseSchema,
  createEvaluationResponseSchema,
  type EvaluationsResponse,
  evaluationsResponseSchema,
  type ManagedEvaluation,
  type StartEvaluationRequest,
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
      source_policy_kind: request.sourcePolicyKind ?? "run",
      source_policy_id: request.sourcePolicyId ?? request.sourceRunId,
      source_run_id: request.sourceRunId,
      source_artifact: request.sourceArtifact,
      preset_id: request.presetId,
      policy_mode: request.policyMode,
    }),
  });
  const payload = parseApiPayload(createEvaluationResponseSchema, await parseJson(response));
  return payload.evaluation;
}

export async function createEvaluationPreset(request: CreateEvaluationPresetRequest) {
  const response = await fetch("/api/evaluation-presets", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      name: request.name,
      seed: request.seed,
      renderer: request.renderer,
      target: {
        mode: request.targetMode,
        course_ids: request.courseIds,
        cup_ids: request.cupIds,
        difficulties: request.difficulties,
        repeats_per_target: request.repeatsPerTarget,
        baseline_variant_count: request.baselineVariantCount,
      },
    }),
  });
  const payload = parseApiPayload(createEvaluationPresetResponseSchema, await parseJson(response));
  return payload.preset;
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

export async function deleteEvaluationPreset(presetId: string): Promise<boolean> {
  const response = await fetch(`/api/evaluation-presets/${encodeURIComponent(presetId)}`, {
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
  throw new Error("run-manager delete evaluation preset response is invalid");
}

export async function startEvaluation(
  evaluationId: string,
  request: StartEvaluationRequest,
): Promise<ManagedEvaluation> {
  const response = await fetch(`/api/evaluations/${encodeURIComponent(evaluationId)}/start`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ device: request.device, worker_count: request.workerCount }),
  });
  const payload = parseApiPayload(startEvaluationResponseSchema, await parseJson(response));
  return payload.evaluation;
}

export async function cancelEvaluation(evaluationId: string): Promise<ManagedEvaluation> {
  const response = await fetch(`/api/evaluations/${encodeURIComponent(evaluationId)}/cancel`, {
    method: "POST",
  });
  const payload = parseApiPayload(cancelEvaluationResponseSchema, await parseJson(response));
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
