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

const rewardConfigSchema = z
  .object({
    time_penalty_per_frame: z.number(),
    reverse_time_penalty_scale: z.number().nonnegative(),
    low_speed_time_penalty_scale: z.number().nonnegative(),
    slow_speed_time_penalty_scale: z.number().nonnegative(),
    slow_speed_time_penalty_start_kph: z.number().nonnegative(),
    slow_speed_time_penalty_power: z.number().positive(),
    progress_bucket_distance: z.number().positive(),
    progress_bucket_reward: z.number().nonnegative(),
    progress_reward_interval_frames: z.number().int().positive(),
    airborne_progress_bucket_distance: z.number().positive().nullable(),
    outside_bounds_reentry_progress_distance_cap: z.number().nonnegative().nullable(),
    airborne_offtrack_penalty_scale: z.number().nonnegative(),
    airborne_offtrack_recovery_reward_scale: z.number().nonnegative(),
    airborne_offtrack_recovery_requires_descending: z.boolean(),
    airborne_offtrack_recovery_descend_epsilon: z.number().nonnegative(),
    lap_completion_bonus: z.number().nonnegative(),
    lap_position_scale: z.number().nonnegative(),
    energy_loss_epsilon: z.number().nonnegative(),
    energy_refill_progress_multiplier: z.number().min(1),
    dirt_progress_multiplier: z.number().nonnegative(),
    ice_progress_multiplier: z.number().nonnegative(),
    dirt_entry_penalty: z.number().max(0),
    ice_entry_penalty: z.number().max(0),
    energy_refill_collision_cooldown_frames: z.number().int().nonnegative(),
    energy_full_refill_lap_bonus: z.number().nonnegative(),
    energy_full_refill_min_gain_fraction: z.number().min(0).max(1),
    gas_underuse_penalty: z.number().max(0),
    gas_underuse_threshold: z.number().min(0).max(1),
    steer_oscillation_penalty: z.number().max(0),
    steer_oscillation_deadzone: z.number().nonnegative(),
    steer_oscillation_cap: z.number().positive(),
    steer_oscillation_power: z.number().positive(),
    manual_boost_reward: z.number().nonnegative(),
    boost_pad_reward: z.number().nonnegative(),
    boost_pad_reward_progress_window: z.number().positive(),
    lean_request_penalty: z.number().max(0),
    lean_low_speed_penalty: z.number().max(0),
    lean_low_speed_penalty_max_speed_kph: z.number().nonnegative(),
    airborne_pitch_up_penalty: z.number().max(0),
    damage_taken_frame_penalty: z.number().max(0),
    damage_taken_streak_ramp_penalty: z.number().max(0),
    damage_taken_streak_cap_frames: z.number().int().nonnegative(),
    airborne_landing_reward: z.number(),
    collision_recoil_penalty: z.number(),
    failure_penalty: z.number(),
    truncation_penalty: z.number(),
    step_reward_clip_min: z.number().nullable(),
    step_reward_clip_max: z.number().nullable(),
  })
  .refine(
    (reward) =>
      reward.step_reward_clip_min === null ||
      reward.step_reward_clip_max === null ||
      reward.step_reward_clip_min <= reward.step_reward_clip_max,
    {
      message: "step_reward_clip_min must be <= step_reward_clip_max",
      path: ["step_reward_clip_min"],
    },
  );

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
