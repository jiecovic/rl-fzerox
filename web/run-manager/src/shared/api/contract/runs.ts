// web/run-manager/src/shared/api/contract/runs.ts
import { z } from "zod";

import { managedRunConfigSchema } from "@/shared/api/contract/config";
import {
  engineSettingModeSchema,
  runCommandSchema,
  runStatusSchema,
  vehicleSelectionModeSchema,
} from "@/shared/api/contract/enums";

const managedRunVehicleSetupSchema = z.object({
  selection_mode: vehicleSelectionModeSchema,
  selected_vehicle_ids: z.array(z.string()).min(1),
  engine_mode: engineSettingModeSchema,
  engine_setting_raw_value: z.number().int().min(0).max(100),
  engine_setting_min_raw_value: z.number().int().min(0).max(100),
  engine_setting_max_raw_value: z.number().int().min(0).max(100),
});

export const managedTemplateSchema = z.object({
  id: z.string(),
  name: z.string(),
  created_at: z.string(),
  updated_at: z.string(),
  config: managedRunConfigSchema,
});

export const managedDraftSchema = z.object({
  id: z.string(),
  name: z.string(),
  source_run_id: z.string().nullable(),
  source_artifact: z.enum(["latest", "best"]).nullable(),
  source_num_timesteps: z.number().int().nonnegative().nullable(),
  created_at: z.string(),
  updated_at: z.string(),
  config: managedRunConfigSchema,
});

export const managedRunSummarySchema = z.object({
  id: z.string(),
  name: z.string(),
  status: runStatusSchema,
  config_hash: z.string(),
  action_repeat: z.number().int().positive(),
  vehicle_setup: managedRunVehicleSetupSchema,
  created_at: z.string(),
  lineage_id: z.string(),
  lineage_groups: z.array(z.string()),
  lineage_step_offset: z.number().int().nonnegative(),
  started_at: z.string().nullable(),
  stopped_at: z.string().nullable(),
  parent_run_id: z.string().nullable(),
  source_run_id: z.string().nullable(),
  source_artifact: z.enum(["latest", "best"]).nullable(),
  source_num_timesteps: z.number().int().nonnegative().nullable(),
  active_alt_baseline_count: z.number().int().nonnegative().default(0),
  pending_command: runCommandSchema.nullable(),
  worker_heartbeat_at: z.string().nullable(),
  runtime: z
    .object({
      total_timesteps: z.number().int().nonnegative(),
      num_timesteps: z.number().int().nonnegative(),
      progress_fraction: z.number().min(0).max(1),
      updated_at: z.string(),
      fps: z.number().nullable(),
      episode_reward_mean: z.number().nullable(),
      episode_length_mean: z.number().nullable(),
      approx_kl: z.number().nullable(),
      entropy_loss: z.number().nullable(),
      value_loss: z.number().nullable(),
      policy_gradient_loss: z.number().nullable(),
    })
    .nullable(),
  recent_events: z.array(
    z.object({
      created_at: z.string(),
      kind: z.string(),
      message: z.string(),
    }),
  ),
});

export const managedRunSchema = managedRunSummarySchema.extend({
  config: managedRunConfigSchema,
});

export const managedRunMetricSampleSchema = z.object({
  run_id: z.string(),
  created_at: z.string(),
  total_timesteps: z.number().int().nonnegative(),
  num_timesteps: z.number().int().nonnegative(),
  lineage_num_timesteps: z.number().int().nonnegative(),
  progress_fraction: z.number().min(0).max(1),
  metrics: z.record(z.string(), z.number()).default({}),
  fps: z.number().nullable(),
  episode_reward_mean: z.number().nullable(),
  episode_length_mean: z.number().nullable(),
  approx_kl: z.number().nullable(),
  entropy_loss: z.number().nullable(),
  value_loss: z.number().nullable(),
  policy_gradient_loss: z.number().nullable(),
});

export const trackSamplingRuntimeEntrySchema = z.object({
  track_id: z.string(),
  course_key: z.string(),
  label: z.string(),
  current_weight: z.number().nonnegative(),
  current_probability: z.number().min(0).max(1),
  episode_count: z.number().int().nonnegative(),
  finished_episode_count: z.number().int().nonnegative(),
  success_sample_count: z.number().int().nonnegative(),
  episode_share: z.number().min(0).max(1),
  success_rate: z.number().min(0).max(1).nullable(),
  generation_episode_count: z.number().int().nonnegative(),
  generation_finished_episode_count: z.number().int().nonnegative(),
  generation_success_sample_count: z.number().int().nonnegative(),
  generation_success_rate: z.number().min(0).max(1).nullable(),
  generation_ema_completion_fraction: z.number().min(0).max(1).nullable(),
  target_step_share: z.number().min(0).max(1),
  completed_frames: z.number().int().nonnegative(),
  completed_env_steps: z.number().int().nonnegative(),
  step_share: z.number().min(0).max(1),
  ema_episode_frames: z.number().nonnegative().nullable(),
  ema_completion_fraction: z.number().min(0).max(1).nullable(),
  generated_course_slot: z.number().int().nonnegative().nullable(),
  generated_course_generation: z.number().int().positive().nullable(),
});

export const trackSamplingRuntimeStateSchema = z.object({
  sampling_mode: z.string(),
  action_repeat: z.number().int().positive(),
  update_episodes: z.number().int().positive(),
  ema_alpha: z.number().gt(0).max(1),
  max_weight_scale: z.number().min(1),
  adaptive_completion_weight: z.number().nonnegative(),
  adaptive_target_completion: z.number().min(0).max(1),
  adaptive_min_confidence_episodes: z.number().int().positive(),
  adaptive_confidence_scale: z.number().min(1),
  update_count: z.number().int().nonnegative(),
  episodes_since_update: z.number().int().nonnegative(),
  entries: z.array(trackSamplingRuntimeEntrySchema),
});

export const runAltBaselineSchema = z.object({
  id: z.string(),
  course_key: z.string(),
  reset_variant_key: z.string(),
  source_entry_id: z.string(),
  label: z.string(),
  state_path: z.string(),
  weight: z.number(),
  created_at: z.string(),
  updated_at: z.string(),
});

export const engineTuningRuntimeCandidateSchema = z.object({
  context_key: z.string(),
  course_key: z.string(),
  vehicle_id: z.string(),
  engine_setting_raw_value: z.number().int().min(0).max(100),
  finish_count: z.number().int().nonnegative(),
  mean_score: z.number().nullable(),
  raw_mean_score: z.number().nullable(),
  best_score: z.number().nullable(),
  best_finish_time_ms: z.number().int().positive().nullable(),
});

export const engineTuningRuntimeCandidateEstimateSchema = z.object({
  engine_setting_raw_value: z.number().int().min(0).max(100),
  selection_probability: z.number().min(0).max(1),
  mean_score: z.number(),
  uncertainty_score: z.number().nonnegative(),
  estimated_finish_time_ms: z.number().int().positive(),
  best_finish_time_ms: z.number().int().positive().nullable(),
  finish_count: z.number().int().nonnegative(),
});

export const engineTuningRuntimeContextSchema = z.object({
  context_key: z.string(),
  course_key: z.string(),
  vehicle_id: z.string(),
  finish_count: z.number().int().nonnegative(),
  observed_candidate_count: z.number().int().nonnegative(),
  model_ready: z.boolean().default(true),
  warmup_successes: z.number().int().nonnegative().default(0),
  warmup_remaining: z.number().int().nonnegative().default(0),
  recommended_engine_setting_raw_value: z.number().int().min(0).max(100),
  candidates: z.array(engineTuningRuntimeCandidateEstimateSchema),
});

export const engineTuningRuntimeStateSchema = z.object({
  version: z.number().int().positive(),
  update_count: z.number().int().nonnegative(),
  model_backend: z.enum(["bandit", "gaussian_process", "mlp_ensemble"]).nullable().default(null),
  candidates: z.array(engineTuningRuntimeCandidateSchema),
  contexts: z.array(engineTuningRuntimeContextSchema).default([]),
});

export const templatesResponseSchema = z.object({
  templates: z.array(managedTemplateSchema),
});

export const draftsResponseSchema = z.object({
  drafts: z.array(managedDraftSchema),
});

export const runsResponseSchema = z.object({
  runs: z.array(managedRunSummarySchema),
});

export const runsLiveMessageSchema = runsResponseSchema.extend({
  type: z.literal("runs_snapshot"),
});

export const runsLiveErrorMessageSchema = z.object({
  type: z.literal("runs_error"),
  message: z.string(),
});

export const runsLiveUpdateSchema = z.discriminatedUnion("type", [
  runsLiveMessageSchema,
  runsLiveErrorMessageSchema,
]);

export const runTrackSamplingLiveMessageSchema = z.object({
  type: z.literal("track_sampling_snapshot"),
  state: trackSamplingRuntimeStateSchema.nullable(),
});

export const runTrackSamplingLiveErrorMessageSchema = z.object({
  type: z.literal("track_sampling_error"),
  message: z.string(),
});

export const runTrackSamplingLiveUpdateSchema = z.discriminatedUnion("type", [
  runTrackSamplingLiveMessageSchema,
  runTrackSamplingLiveErrorMessageSchema,
]);

export const runResponseSchema = z.object({
  run: managedRunSchema,
});

export const createDraftResponseSchema = z.object({
  draft: managedDraftSchema,
});

export const createRunResponseSchema = z.object({
  run: managedRunSchema,
});

export const forkRunResponseSchema = createRunResponseSchema;

export const deleteRunResponseSchema = z.object({
  deleted: z.boolean(),
});

export const resetTrackSamplingResponseSchema = z.object({
  reset: z.boolean(),
});

export const resetEngineTuningResponseSchema = z.object({
  cleared: z.number().int().nonnegative(),
  reset: z.boolean(),
  run: managedRunSchema,
});

export const clearAltBaselinesResponseSchema = z.object({
  cleared: z.number().int().nonnegative(),
  run: managedRunSchema,
});

export const openRunDirectoryResponseSchema = z.object({
  opened: z.boolean(),
});

export const watchRunResponseSchema = z.object({
  status: z.enum(["started", "already_running"]),
});

export const runMetricsResponseSchema = z.object({
  samples: z.array(managedRunMetricSampleSchema),
});

export const runTrackSamplingResponseSchema = z.object({
  state: trackSamplingRuntimeStateSchema.nullable(),
});

export const runAltBaselinesResponseSchema = z.object({
  baselines: z.array(runAltBaselineSchema),
});

export const runEngineTuningResponseSchema = z.object({
  enabled: z.boolean(),
  state: engineTuningRuntimeStateSchema.nullable(),
});

export const tensorboardViewGroupSchema = z.object({
  name: z.string(),
  slug: z.string(),
  path: z.string(),
  lineage_count: z.number().int().nonnegative(),
  run_count: z.number().int().nonnegative(),
});

export const updateLineageGroupsResponseSchema = z.object({
  lineage_id: z.string(),
  lineage_groups: z.array(z.string()),
  tensorboard_views: z.array(tensorboardViewGroupSchema),
});

export const rebuildTensorboardViewsResponseSchema = z.object({
  tensorboard_views: z.array(tensorboardViewGroupSchema),
});

export type ManagedTemplate = z.infer<typeof managedTemplateSchema>;
export type ManagedDraft = z.infer<typeof managedDraftSchema>;
export type ManagedRun = z.infer<typeof managedRunSummarySchema>;
export type ManagedRunDetail = z.infer<typeof managedRunSchema>;
export type ManagedRunMetricSample = z.infer<typeof managedRunMetricSampleSchema>;
export type RunAltBaseline = z.infer<typeof runAltBaselineSchema>;
export type RunsLiveUpdate = z.infer<typeof runsLiveUpdateSchema>;
export type RunTrackSamplingLiveUpdate = z.infer<typeof runTrackSamplingLiveUpdateSchema>;
export type EngineTuningRuntimeCandidate = z.infer<typeof engineTuningRuntimeCandidateSchema>;
export type EngineTuningRuntimeCandidateEstimate = z.infer<
  typeof engineTuningRuntimeCandidateEstimateSchema
>;
export type EngineTuningRuntimeContext = z.infer<typeof engineTuningRuntimeContextSchema>;
export type EngineTuningRuntimeState = z.infer<typeof engineTuningRuntimeStateSchema>;
export type TrackSamplingRuntimeEntry = z.infer<typeof trackSamplingRuntimeEntrySchema>;
export type TrackSamplingRuntimeState = z.infer<typeof trackSamplingRuntimeStateSchema>;
export type TensorboardViewGroup = z.infer<typeof tensorboardViewGroupSchema>;
