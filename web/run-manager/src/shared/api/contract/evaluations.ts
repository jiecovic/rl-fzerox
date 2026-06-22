// web/run-manager/src/shared/api/contract/evaluations.ts
import { z } from "zod";

import {
  policyPlaybackModeSchema,
  type savePolicyArtifactSchema,
} from "@/shared/api/contract/saveGames";

export const evaluationStatusSchema = z.enum([
  "created",
  "running",
  "completed",
  "failed",
  "cancelled",
]);

export const evaluationModeSchema = z.enum(["time_attack", "gp_cup", "career_target", "best_of"]);

export const evaluationTargetSpecSchema = z.object({
  mode: evaluationModeSchema,
  course_ids: z.array(z.string()),
  cup_ids: z.array(z.string()),
  difficulties: z.array(z.string()),
  vehicle_ids: z.array(z.string()),
  repeats_per_target: z.number().int().positive(),
});

export const evaluationCheckpointSnapshotSchema = z.object({
  source_run_id: z.string().nullable(),
  source_run_name: z.string().nullable(),
  artifact: z.enum(["latest", "best", "final"]),
  source_policy_path: z.string(),
  copied_policy_path: z.string(),
  source_model_path: z.string().nullable(),
  copied_model_path: z.string().nullable(),
  local_num_timesteps: z.number().int().nonnegative().nullable(),
  lineage_num_timesteps: z.number().int().nonnegative().nullable(),
  source_mtime_ns: z.number().int().nonnegative().nullable(),
});

export const managedEvaluationSchema = z.object({
  id: z.string(),
  name: z.string(),
  status: evaluationStatusSchema,
  evaluation_dir: z.string(),
  source_run_id: z.string().nullable(),
  source_artifact: z.enum(["latest", "best", "final"]).nullable(),
  policy_mode: policyPlaybackModeSchema,
  seed: z.number().int().nonnegative(),
  target: evaluationTargetSpecSchema,
  checkpoint: evaluationCheckpointSnapshotSchema,
  created_at: z.string(),
  updated_at: z.string(),
  started_at: z.string().nullable(),
  finished_at: z.string().nullable(),
  result_json_path: z.string().nullable(),
  error_message: z.string().nullable(),
});

export const evaluationsResponseSchema = z.object({
  evaluations: z.array(managedEvaluationSchema),
});

export const createEvaluationResponseSchema = z.object({
  evaluation: managedEvaluationSchema,
});

export type EvaluationMode = z.infer<typeof evaluationModeSchema>;
export type ManagedEvaluation = z.infer<typeof managedEvaluationSchema>;
export type EvaluationStatus = z.infer<typeof evaluationStatusSchema>;
export type EvaluationSourceArtifact = z.infer<typeof savePolicyArtifactSchema>;

export interface CreateEvaluationRequest {
  courseIds: readonly string[];
  cupIds: readonly string[];
  difficulties: readonly string[];
  name: string;
  policyMode: "deterministic" | "stochastic";
  repeatsPerTarget: number;
  seed: number;
  sourceArtifact: EvaluationSourceArtifact;
  sourceRunId: string;
  targetMode: EvaluationMode;
  vehicleIds: readonly string[];
}
