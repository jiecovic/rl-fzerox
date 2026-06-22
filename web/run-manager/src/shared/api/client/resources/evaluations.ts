// web/run-manager/src/shared/api/client/resources/evaluations.ts
import { getJson, parseApiPayload, parseJson } from "@/shared/api/client/http";
import {
  type CreateEvaluationRequest,
  createEvaluationResponseSchema,
  evaluationsResponseSchema,
  type ManagedEvaluation,
} from "@/shared/api/contract";

export async function fetchEvaluations(): Promise<ManagedEvaluation[]> {
  const payload = parseApiPayload(evaluationsResponseSchema, await getJson("/api/evaluations"));
  return payload.evaluations;
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
      source_artifact: request.sourceArtifact,
      policy_mode: request.policyMode,
      seed: request.seed,
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
