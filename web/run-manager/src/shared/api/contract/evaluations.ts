// web/run-manager/src/shared/api/contract/evaluations.ts
import { z } from "zod";

import { managedRunConfigSchema } from "@/shared/api/contract/config";
import type { WatchDevice } from "@/shared/api/contract/enums";
import {
  policyPlaybackModeSchema,
  type SavePolicySourceKind,
  savePolicySourceKindSchema,
} from "@/shared/api/contract/saveGames";

export const evaluationStatusSchema = z.enum([
  "created",
  "running",
  "cancelling",
  "completed",
  "failed",
  "cancelled",
]);

export const evaluationModeSchema = z.enum(["time_attack_course", "gp_course"]);
export const evaluationSourceArtifactSchema = z.enum(["latest", "best"]);
export const evaluationBaselineSuiteStatusSchema = z.enum(["not_created", "ready", "failed"]);

export const evaluationTargetSpecSchema = z.object({
  mode: evaluationModeSchema,
  course_ids: z.array(z.string()),
  cup_ids: z.array(z.string()),
  difficulties: z.array(z.string()),
  vehicle_ids: z.array(z.string()),
  repeats_per_target: z.number().int().positive(),
  baseline_variant_count: z.number().int().min(1).max(16),
});

export const evaluationCheckpointSnapshotSchema = z.object({
  source_run_id: z.string().nullable(),
  source_run_name: z.string().nullable(),
  artifact: evaluationSourceArtifactSchema,
  source_policy_path: z.string(),
  copied_policy_path: z.string(),
  source_model_path: z.string().nullable(),
  copied_model_path: z.string().nullable(),
  local_num_timesteps: z.number().int().nonnegative().nullable(),
  lineage_num_timesteps: z.number().int().nonnegative().nullable(),
  source_mtime_ns: z.string().regex(/^\d+$/).nullable(),
});

export const evaluationProgressSchema = z.object({
  completed_attempts: z.number().int().nonnegative(),
  total_attempts: z.number().int().nonnegative().nullable(),
  result_status: z.string().nullable(),
});

export const evaluationMetricSummarySchema = z.object({
  key: z.string(),
  label: z.string(),
  attempt_count: z.number().int().nonnegative(),
  success_count: z.number().int().nonnegative(),
  success_rate: z.number().nullable(),
  finish_count: z.number().int().nonnegative(),
  finish_rate: z.number().nullable(),
  completion_rate: z.number().nullable(),
  mean_finish_time_ms: z.number().nullable(),
  best_finish_time_ms: z.number().nullable(),
  mean_position: z.number().nullable(),
  best_position: z.number().int().nullable(),
  total_env_steps: z.number().int().nonnegative(),
  mean_episode_length_steps: z.number().nullable(),
  mean_episode_return: z.number().nullable(),
  best_episode_return: z.number().nullable(),
  average_speed: z.number().nullable(),
});

export const evaluationAttemptSummarySchema = z.object({
  attempt_id: z.string(),
  target_id: z.string(),
  target_label: z.string().nullable(),
  status: z.string(),
  seed: z.string().regex(/^\d+$/).nullable(),
  cup_id: z.string().nullable(),
  difficulty: z.string().nullable(),
  vehicle_id: z.string().nullable(),
  total_race_time_ms: z.number().int().nullable(),
  env_steps: z.number().int().nullable(),
  episode_return: z.number().nullable(),
  position: z.number().int().nullable(),
  completion_ratio: z.number().nullable(),
  closed_at_utc: z.string().nullable(),
});

export const evaluationRuntimeSchema = z.object({
  device: z.enum(["cpu", "cuda"]),
  worker_count: z.number().int().positive(),
});

export const evaluationResultSummarySchema = z.object({
  status: z.string(),
  started_at_utc: z.string().nullable(),
  closed_at_utc: z.string().nullable(),
  runtime: evaluationRuntimeSchema.nullable(),
  overall: evaluationMetricSummarySchema.nullable(),
  courses: z.array(evaluationMetricSummarySchema),
  attempts: z.array(evaluationAttemptSummarySchema),
});

export const evaluationBaselineSuiteSchema = z.object({
  id: z.string(),
  preset_id: z.string(),
  preset_version: z.number().int().positive(),
  status: evaluationBaselineSuiteStatusSchema,
  suite_dir: z.string(),
  manifest_path: z.string().nullable(),
  error_message: z.string().nullable(),
  created_at: z.string().nullable(),
  updated_at: z.string().nullable(),
  materialized_at: z.string().nullable(),
});

export const managedEvaluationSchema = z.object({
  id: z.string(),
  name: z.string(),
  status: evaluationStatusSchema,
  evaluation_dir: z.string(),
  source_policy_kind: savePolicySourceKindSchema,
  source_policy_id: z.string().nullable(),
  source_run_id: z.string().nullable(),
  source_artifact: evaluationSourceArtifactSchema.nullable(),
  preset_id: z.string(),
  preset_version: z.number().int().positive(),
  policy_mode: policyPlaybackModeSchema,
  seed: z.number().int().nonnegative(),
  target: evaluationTargetSpecSchema,
  config: managedRunConfigSchema,
  checkpoint: evaluationCheckpointSnapshotSchema,
  created_at: z.string(),
  updated_at: z.string(),
  started_at: z.string().nullable(),
  finished_at: z.string().nullable(),
  result_json_path: z.string().nullable(),
  error_message: z.string().nullable(),
  progress: evaluationProgressSchema,
  result_summary: evaluationResultSummarySchema.nullable(),
  baseline_suite: evaluationBaselineSuiteSchema,
});

export const managedEvaluationPresetSchema = z.object({
  id: z.string(),
  name: z.string(),
  version: z.number().int().positive(),
  seed: z.number().int().nonnegative(),
  renderer: z.enum(["angrylion", "gliden64"]),
  target: evaluationTargetSpecSchema,
  builtin: z.boolean(),
  created_at: z.string(),
  updated_at: z.string(),
});

export const evaluationsResponseSchema = z.object({
  evaluations: z.array(managedEvaluationSchema),
  presets: z.array(managedEvaluationPresetSchema),
  baseline_suites: z.array(evaluationBaselineSuiteSchema),
});

export const createEvaluationResponseSchema = z.object({
  evaluation: managedEvaluationSchema,
});

export const createEvaluationPresetResponseSchema = z.object({
  preset: managedEvaluationPresetSchema,
});

export const startEvaluationResponseSchema = z.object({
  evaluation: managedEvaluationSchema,
});

export const cancelEvaluationResponseSchema = z.object({
  evaluation: managedEvaluationSchema,
});

export const updateEvaluationResponseSchema = z.object({
  evaluation: managedEvaluationSchema,
});

export type EvaluationMode = z.infer<typeof evaluationModeSchema>;
export type ManagedEvaluation = z.infer<typeof managedEvaluationSchema>;
export type ManagedEvaluationPreset = z.infer<typeof managedEvaluationPresetSchema>;
export type EvaluationBaselineSuite = z.infer<typeof evaluationBaselineSuiteSchema>;
export type EvaluationBaselineSuiteStatus = z.infer<typeof evaluationBaselineSuiteStatusSchema>;
export type EvaluationMetricSummary = z.infer<typeof evaluationMetricSummarySchema>;
export type EvaluationAttemptSummary = z.infer<typeof evaluationAttemptSummarySchema>;
export type EvaluationResultSummary = z.infer<typeof evaluationResultSummarySchema>;
export type EvaluationRuntime = z.infer<typeof evaluationRuntimeSchema>;
export type EvaluationsResponse = z.infer<typeof evaluationsResponseSchema>;
export type EvaluationStatus = z.infer<typeof evaluationStatusSchema>;
export type EvaluationSourceArtifact = z.infer<typeof evaluationSourceArtifactSchema>;

export interface CreateEvaluationRequest {
  name: string;
  policyMode: "deterministic" | "stochastic";
  presetId: string;
  sourceArtifact: EvaluationSourceArtifact;
  sourcePolicyKind?: SavePolicySourceKind;
  sourcePolicyId?: string | null;
  sourceRunId?: string | null;
}

export interface StartEvaluationRequest {
  device: WatchDevice;
  workerCount: number;
}

export interface CreateEvaluationPresetRequest {
  courseIds: readonly string[];
  cupIds: readonly string[];
  difficulties: readonly string[];
  name: string;
  renderer: "angrylion" | "gliden64";
  baselineVariantCount: number;
  repeatsPerTarget: number;
  seed: number;
  targetMode: EvaluationMode;
}

export interface UpdateEvaluationRequest {
  name: string;
}
