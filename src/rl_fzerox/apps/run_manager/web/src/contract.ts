import { z } from "zod";

const runStatusSchema = z.enum(["created", "running", "paused", "stopped", "finished", "failed"]);

const trainConfigSchema = z.object({
  algorithm: z.literal("maskable_hybrid_recurrent_ppo"),
  num_envs: z.number().int().positive(),
  total_timesteps: z.number().int().positive(),
  n_steps: z.number().int().positive(),
  n_epochs: z.number().int().positive(),
  batch_size: z.number().int().positive(),
  learning_rate: z.number().positive(),
  gamma: z.number().positive().max(1),
  gae_lambda: z.number().positive().max(1),
  clip_range: z.number().positive(),
  clip_range_vf: z.number().positive().nullable(),
  ent_coef: z.number().nonnegative(),
  vf_coef: z.number().positive(),
  max_grad_norm: z.number().positive(),
  normalize_advantage: z.boolean(),
  target_kl: z.number().positive().nullable(),
  stats_window_size: z.number().int().positive(),
  course_context_dropout_prob: z.number().min(0).max(1),
});

const observationConfigSchema = z.object({
  preset: z.literal("crop_60x76"),
  frame_stack: z.number().int().positive().max(8),
  stack_mode: z.enum(["rgb", "gray", "luma_chroma"]),
  minimap_layer: z.boolean(),
  progress_source: z.enum(["lap_progress", "segment_progress", "none"]),
  zero_edge_ratio: z.boolean(),
  zero_outside_track_bounds: z.boolean(),
});

const policyConfigSchema = z.object({
  conv_profile: z.enum(["nature", "nature_32_64_128", "nature_wide"]),
  fusion_features_dim: z.number().int().positive(),
  recurrent_hidden_size: z.number().int().positive(),
  pi_net_arch: z.array(z.number().int().positive()),
  vf_net_arch: z.array(z.number().int().positive()),
  gas_on_logit: z.number(),
});

const rewardConfigSchema = z.object({
  manual_boost_reward: z.number().nonnegative(),
  boost_pad_reward: z.number().nonnegative(),
  lean_request_penalty: z.number().max(0),
  lean_low_speed_penalty: z.number().max(0),
  lean_low_speed_penalty_max_speed_kph: z.number().nonnegative(),
  airborne_pitch_up_penalty: z.number().max(0),
  collision_recoil_penalty: z.number(),
  failure_penalty: z.number(),
  truncation_penalty: z.number(),
});

export const managedRunConfigSchema = z.object({
  version: z.literal(1),
  seed: z.number().int(),
  preset_name: z.string(),
  train: trainConfigSchema,
  observation: observationConfigSchema,
  policy: policyConfigSchema,
  reward: rewardConfigSchema,
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
  created_at: z.string(),
  updated_at: z.string(),
  config: managedRunConfigSchema,
});

export const managedRunSchema = z.object({
  id: z.string(),
  name: z.string(),
  status: runStatusSchema,
  created_at: z.string(),
  started_at: z.string().nullable(),
  stopped_at: z.string().nullable(),
  parent_run_id: z.string().nullable(),
  source_run_id: z.string().nullable(),
  source_artifact: z.string().nullable(),
  config: managedRunConfigSchema,
});

export const templatesResponseSchema = z.object({
  templates: z.array(managedTemplateSchema),
});

export const draftsResponseSchema = z.object({
  drafts: z.array(managedDraftSchema),
});

export const runsResponseSchema = z.object({
  runs: z.array(managedRunSchema),
});

export const createDraftResponseSchema = z.object({
  draft: managedDraftSchema,
});

export type ManagedRunConfig = z.infer<typeof managedRunConfigSchema>;
export type ManagedTemplate = z.infer<typeof managedTemplateSchema>;
export type ManagedDraft = z.infer<typeof managedDraftSchema>;
export type ManagedRun = z.infer<typeof managedRunSchema>;
