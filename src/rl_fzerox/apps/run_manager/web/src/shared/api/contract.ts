import { z } from "zod";

const runStatusSchema = z.enum(["created", "running", "paused", "stopped", "finished", "failed"]);
const observationPresetSchema = z.enum([
  "crop_84x116",
  "crop_92x124",
  "crop_116x164",
  "crop_98x130",
  "crop_66x82",
  "crop_60x76",
  "crop_68x68",
  "crop_84x84",
  "crop_76x100",
  "crop_64x64",
]);
const stateComponentNameSchema = z.enum([
  "vehicle_state",
  "machine_context",
  "track_position",
  "surface_state",
  "course_context",
  "control_history",
]);
const stateComponentModeSchema = z.enum(["include", "zero", "exclude"]);
const actionHistoryControlSchema = z.enum([
  "steer",
  "gas",
  "thrust",
  "air_brake",
  "boost",
  "lean",
  "pitch",
]);
const convProfileSchema = z.enum([
  "auto",
  "nature",
  "nature_32_64_128",
  "nature_wide",
  "nature_extra_k3",
  "compact_deep",
  "compact_bottleneck",
  "tiny_256",
]);

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

const stateComponentConfigSchema = z.object({
  name: stateComponentNameSchema,
  mode: stateComponentModeSchema,
  encoding: z.enum(["none", "one_hot_builtin"]).nullable(),
  progress_source: z.enum(["lap_progress", "segment_progress", "none"]).nullable(),
  length: z.number().int().positive().max(16).nullable(),
  controls: z.array(actionHistoryControlSchema).nullable(),
});

const stateFeatureConfigSchema = z.object({
  name: z.string(),
  mode: stateComponentModeSchema,
});

const observationConfigSchema = z.object({
  preset: observationPresetSchema,
  frame_stack: z.number().int().positive().max(8),
  stack_mode: z.enum(["rgb", "gray", "luma_chroma"]),
  minimap_layer: z.boolean(),
  resize_filter: z.enum(["nearest", "bilinear"]),
  minimap_resize_filter: z.enum(["nearest", "bilinear"]),
  state_components: z.array(stateComponentConfigSchema),
  state_feature_modes: z.array(stateFeatureConfigSchema),
});

const policyConfigSchema = z.object({
  conv_profile: convProfileSchema,
  features_dim: z.union([z.literal("auto"), z.number().int().positive()]),
  state_features_dim: z.number().int().positive(),
  fusion_features_dim: z.number().int().positive(),
  layer_norm: z.boolean(),
  activation: z.enum(["relu", "gelu", "tanh"]).default("relu"),
  recurrent_enabled: z.boolean(),
  recurrent_hidden_size: z.number().int().positive(),
  recurrent_n_lstm_layers: z.number().int().positive(),
  recurrent_shared_lstm: z.boolean(),
  recurrent_enable_critic_lstm: z.boolean(),
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

const selectOptionSchema = z.object({
  value: z.string(),
  label: z.string(),
});

const observationPresetInfoSchema = z.object({
  value: observationPresetSchema,
  label: z.string(),
  height: z.number().int().positive(),
  width: z.number().int().positive(),
});

const stateComponentInfoSchema = z.object({
  name: z.string(),
  low: z.number(),
  high: z.number(),
});

const stateComponentSchema = z.object({
  name: stateComponentNameSchema,
  label: z.string(),
  default_mode: stateComponentModeSchema,
  features: z.array(stateComponentInfoSchema),
});

export const configMetadataSchema = z.object({
  observation_presets: z.array(observationPresetInfoSchema),
  stack_modes: z.array(selectOptionSchema),
  resize_filters: z.array(selectOptionSchema),
  progress_sources: z.array(selectOptionSchema),
  component_modes: z.array(selectOptionSchema),
  action_history_controls: z.array(selectOptionSchema),
  state_components: z.array(stateComponentSchema),
  conv_profiles: z.array(selectOptionSchema),
  activation_functions: z.array(selectOptionSchema).default([
    { value: "relu", label: "relu" },
    { value: "gelu", label: "gelu" },
    { value: "tanh", label: "tanh" },
  ]),
  net_arch_presets: z.array(selectOptionSchema),
});

const shapePreviewSchema = z.object({
  height: z.number().int().positive(),
  width: z.number().int().positive(),
  channels: z.number().int().positive(),
});

const stateFeaturePreviewSchema = z.object({
  component: stateComponentNameSchema,
  name: z.string(),
  mode: stateComponentModeSchema,
});

const convLayerPreviewSchema = z.object({
  name: z.string(),
  in_channels: z.number().int().nonnegative(),
  out_channels: z.number().int().positive(),
  kernel_size: z.number().int().positive(),
  stride: z.number().int().positive(),
  output_height: z.number().int().positive(),
  output_width: z.number().int().positive(),
  params: z.number().int().nonnegative(),
});

const parameterGroupPreviewSchema = z.object({
  name: z.string(),
  params: z.number().int().nonnegative(),
});

const architectureNodePreviewSchema = z.object({
  id: z.string(),
  label: z.string(),
  detail: z.string(),
  tone: z.string(),
});

const architectureLanePreviewSchema = z.object({
  id: z.string(),
  label: z.string(),
  nodes: z.array(architectureNodePreviewSchema),
});

export const policyArchitecturePreviewSchema = z.object({
  image_shape: shapePreviewSchema,
  state_dim: z.number().int().nonnegative(),
  state_features: z.array(stateFeaturePreviewSchema),
  conv_layers: z.array(convLayerPreviewSchema),
  flatten_dim: z.number().int().nonnegative(),
  image_features_dim: z.number().int().nonnegative(),
  state_features_dim: z.number().int().nonnegative(),
  fusion_input_dim: z.number().int().nonnegative(),
  extractor_output_dim: z.number().int().nonnegative(),
  policy_input_dim: z.number().int().nonnegative(),
  parameter_groups: z.array(parameterGroupPreviewSchema),
  total_params: z.number().int().nonnegative(),
  architecture_lanes: z.array(architectureLanePreviewSchema),
});

export type ManagedRunConfig = z.infer<typeof managedRunConfigSchema>;
export type ManagedTemplate = z.infer<typeof managedTemplateSchema>;
export type ManagedDraft = z.infer<typeof managedDraftSchema>;
export type ManagedRun = z.infer<typeof managedRunSchema>;
export type ConfigMetadata = z.infer<typeof configMetadataSchema>;
export type PolicyArchitecturePreview = z.infer<typeof policyArchitecturePreviewSchema>;
export type StateComponentConfig = z.infer<typeof stateComponentConfigSchema>;
export type StateFeatureConfig = z.infer<typeof stateFeatureConfigSchema>;
