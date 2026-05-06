import { z } from "zod";

const runStatusSchema = z.enum(["created", "running", "paused", "stopped", "finished", "failed"]);
const runCommandSchema = z.enum(["pause", "stop"]);
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
const trackPoolModeSchema = z.enum(["built_in", "x_cup"]);
const raceModeSchema = z.enum(["time_attack", "gp_race"]);
const trackSamplingModeSchema = z.enum(["equal", "step_balanced"]);
const vehicleSelectionModeSchema = z.enum(["fixed", "pool"]);
const engineSettingModeSchema = z.enum(["fixed", "random_range"]);
const actionAxisModeSchema = z.enum(["continuous", "discrete"]);
const actionDriveModeSchema = z.enum(["pwm", "on_off"]);
const leanOutputModeSchema = z.enum(["three_way", "independent_buttons"]);
const leanModeSchema = z.enum(["minimum_hold", "release_cooldown", "timer_assist", "raw"]);
const rendererSchema = z.enum(["angrylion", "gliden64"]);
const stateComponentNameSchema = z.enum([
  "vehicle_state",
  "machine_context",
  "track_position",
  "surface_state",
  "course_context",
  "control_history",
]);
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
  "custom",
]);

const customConvLayerSchema = z.object({
  out_channels: z.number().int().positive(),
  kernel_size: z.number().int().positive(),
  stride: z.number().int().positive(),
  padding: z.number().int().nonnegative(),
});

const trainConfigSchema = z.object({
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
  checkpoint_every_rollouts: z.number().int().positive(),
  save_latest_checkpoint: z.boolean(),
  save_best_checkpoint: z.boolean(),
  save_recent_checkpoints: z.boolean(),
  recent_checkpoint_limit: z.number().int().positive().nullable(),
});

const tracksConfigSchema = z.object({
  pool_mode: trackPoolModeSchema,
  race_mode: raceModeSchema,
  sampling_mode: trackSamplingModeSchema,
  selected_course_ids: z.array(z.string()),
});

const vehicleConfigSchema = z
  .object({
    selection_mode: vehicleSelectionModeSchema,
    selected_vehicle_ids: z.array(z.string()).min(1),
    engine_mode: engineSettingModeSchema,
    engine_setting_raw_value: z.number().int().min(0).max(100),
    engine_setting_min_raw_value: z.number().int().min(0).max(100),
    engine_setting_max_raw_value: z.number().int().min(0).max(100),
  })
  .refine(
    (vehicle) => vehicle.engine_setting_min_raw_value <= vehicle.engine_setting_max_raw_value,
    {
      message: "engine_setting_min_raw_value must be <= engine_setting_max_raw_value",
      path: ["engine_setting_min_raw_value"],
    },
  );

const actionConfigSchema = z
  .object({
    action_repeat: z.number().int().positive(),
    steering_mode: actionAxisModeSchema,
    steer_buckets: z.number().int().min(3),
    drive_mode: actionDriveModeSchema,
    force_full_throttle: z.boolean(),
    continuous_drive_deadzone: z.number().min(0).lt(1),
    continuous_drive_full_threshold: z.number().gt(0).max(1),
    continuous_drive_min_thrust: z.number().min(0).max(1),
    include_air_brake: z.boolean(),
    air_brake_mode: actionDriveModeSchema,
    enable_air_brake: z.boolean(),
    mask_air_brake_on_ground: z.boolean(),
    continuous_air_brake_deadzone: z.number().min(0).lt(1),
    continuous_air_brake_full_threshold: z.number().gt(0).max(1),
    continuous_air_brake_min_duty: z.number().min(0).max(1),
    include_boost: z.boolean(),
    enable_boost: z.boolean(),
    boost_unmask_max_speed_kph: z.number().nonnegative().nullable(),
    boost_min_energy_fraction: z.number().min(0).max(1),
    include_lean: z.boolean(),
    enable_lean: z.boolean(),
    lean_output_mode: leanOutputModeSchema,
    lean_mode: leanModeSchema,
    lean_unmask_min_speed_kph: z.number().nonnegative().nullable(),
    lean_initial_lockout_frames: z.number().int().nonnegative(),
    include_pitch: z.boolean(),
    enable_pitch: z.boolean(),
    pitch_mode: actionAxisModeSchema,
    pitch_buckets: z.number().int().min(3),
  })
  .refine((action) => action.steer_buckets % 2 === 1, {
    message: "steer_buckets must be odd",
    path: ["steer_buckets"],
  })
  .refine((action) => action.pitch_buckets % 2 === 1, {
    message: "pitch_buckets must be odd",
    path: ["pitch_buckets"],
  })
  .refine((action) => action.continuous_drive_deadzone < action.continuous_drive_full_threshold, {
    message: "continuous_drive_deadzone must be lower than continuous_drive_full_threshold",
    path: ["continuous_drive_deadzone"],
  })
  .refine(
    (action) => action.continuous_air_brake_deadzone < action.continuous_air_brake_full_threshold,
    {
      message:
        "continuous_air_brake_deadzone must be lower than continuous_air_brake_full_threshold",
      path: ["continuous_air_brake_deadzone"],
    },
  );

const environmentConfigSchema = z.object({
  max_episode_steps: z.number().int().positive(),
  progress_frontier_stall_limit_frames: z.number().int().positive().nullable(),
  progress_frontier_epsilon: z.number().nonnegative(),
  renderer: rendererSchema,
});

const stateComponentConfigSchema = z.object({
  name: stateComponentNameSchema,
  encoding: z.enum(["none", "one_hot_builtin"]).nullable(),
  progress_source: z.enum(["lap_progress", "segment_progress", "none"]).nullable(),
  length: z.number().int().positive().max(16).nullable(),
  controls: z.array(actionHistoryControlSchema).nullable(),
});

const stateFeatureDropoutConfigSchema = z.object({
  name: z.string(),
  dropout_prob: z.number().min(0).max(1),
});

const observationConfigSchema = z.object({
  preset: observationPresetSchema,
  frame_stack: z.number().int().positive().max(8),
  stack_mode: z.enum(["rgb", "gray", "luma_chroma"]),
  minimap_layer: z.boolean(),
  resize_filter: z.enum(["nearest", "bilinear"]),
  minimap_resize_filter: z.enum(["nearest", "bilinear"]),
  state_components: z.array(stateComponentConfigSchema),
  state_feature_dropouts: z.array(stateFeatureDropoutConfigSchema),
});

const policyConfigSchema = z.object({
  conv_profile: convProfileSchema,
  custom_conv_layers: z.array(customConvLayerSchema),
  features_dim: z.union([z.literal("auto"), z.number().int().positive()]),
  state_net_arch: z.array(z.number().int().positive()),
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
    slow_speed_time_penalty_scale: z.number().nonnegative(),
    slow_speed_time_penalty_start_kph: z.number().nonnegative(),
    slow_speed_time_penalty_power: z.number().positive(),
    progress_bucket_distance: z.number().positive(),
    progress_bucket_reward: z.number().nonnegative(),
    progress_reward_interval_frames: z.number().int().positive(),
    suspend_progress_while_outside_track_bounds: z.boolean(),
    outside_bounds_reentry_progress_distance_cap: z.number().nonnegative().nullable(),
    outside_track_frame_penalty: z.number().max(0),
    lap_completion_bonus: z.number().nonnegative(),
    lap_position_scale: z.number().nonnegative(),
    energy_loss_epsilon: z.number().nonnegative(),
    energy_refill_progress_multiplier: z.number().min(1),
    dirt_progress_multiplier: z.number().nonnegative(),
    ice_progress_multiplier: z.number().nonnegative(),
    dirt_entry_penalty: z.number().max(0),
    ice_entry_penalty: z.number().max(0),
    energy_refill_collision_cooldown_frames: z.number().int().nonnegative(),
    air_brake_request_penalty: z.number().max(0),
    manual_boost_reward: z.number().nonnegative(),
    boost_pad_reward: z.number().nonnegative(),
    boost_pad_reward_progress_window: z.number().positive(),
    lean_request_penalty: z.number().max(0),
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
  tracks: tracksConfigSchema,
  vehicle: vehicleConfigSchema,
  action: actionConfigSchema,
  environment: environmentConfigSchema,
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
  source_run_id: z.string().nullable(),
  source_artifact: z.enum(["latest", "best"]).nullable(),
  source_num_timesteps: z.number().int().nonnegative().nullable(),
  created_at: z.string(),
  updated_at: z.string(),
  config: managedRunConfigSchema,
});

export const managedRunSchema = z.object({
  id: z.string(),
  name: z.string(),
  status: runStatusSchema,
  created_at: z.string(),
  lineage_id: z.string(),
  lineage_step_offset: z.number().int().nonnegative(),
  started_at: z.string().nullable(),
  stopped_at: z.string().nullable(),
  parent_run_id: z.string().nullable(),
  source_run_id: z.string().nullable(),
  source_artifact: z.enum(["latest", "best"]).nullable(),
  source_num_timesteps: z.number().int().nonnegative().nullable(),
  pending_command: runCommandSchema.nullable(),
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
  completed_frames: z.number().int().nonnegative(),
  completed_env_steps: z.number().int().nonnegative(),
  step_share: z.number().min(0).max(1),
});

export const trackSamplingRuntimeStateSchema = z.object({
  sampling_mode: z.string(),
  action_repeat: z.number().int().positive(),
  update_episodes: z.number().int().positive(),
  update_count: z.number().int().nonnegative(),
  episodes_since_update: z.number().int().nonnegative(),
  entries: z.array(trackSamplingRuntimeEntrySchema),
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

const trackCupInfoSchema = z.object({
  id: z.string(),
  label: z.string(),
  order: z.number().int().nonnegative(),
  course_ids: z.array(z.string()),
});

const builtInCourseInfoSchema = z.object({
  id: z.string(),
  ref: z.string(),
  display_name: z.string(),
  cup: z.string(),
  cup_label: z.string(),
  course_index: z.number().int().nonnegative(),
  default_selected: z.boolean(),
});

const vehicleInfoSchema = z.object({
  id: z.string(),
  display_name: z.string(),
  character_index: z.number().int().nonnegative(),
  machine_select_slot: z.number().int().nonnegative(),
  menu_row: z.number().int().nonnegative(),
  menu_column: z.number().int().nonnegative(),
});

const engineSettingPresetInfoSchema = z.object({
  id: z.string(),
  display_name: z.string(),
  raw_value: z.number().int().min(0).max(100),
});

const stateComponentInfoSchema = z.object({
  name: z.string(),
  low: z.number(),
  high: z.number(),
});

const stateComponentSchema = z.object({
  name: stateComponentNameSchema,
  label: z.string(),
  features: z.array(stateComponentInfoSchema),
});

export const configMetadataSchema = z.object({
  observation_presets: z.array(observationPresetInfoSchema),
  track_pool_modes: z.array(selectOptionSchema),
  race_modes: z.array(selectOptionSchema),
  track_sampling_modes: z.array(selectOptionSchema),
  track_cups: z.array(trackCupInfoSchema),
  built_in_courses: z.array(builtInCourseInfoSchema),
  vehicles: z.array(vehicleInfoSchema),
  engine_setting_presets: z.array(engineSettingPresetInfoSchema),
  steering_modes: z.array(selectOptionSchema),
  drive_modes: z.array(selectOptionSchema),
  lean_output_modes: z.array(selectOptionSchema),
  lean_modes: z.array(selectOptionSchema),
  stack_modes: z.array(selectOptionSchema),
  resize_filters: z.array(selectOptionSchema),
  progress_sources: z.array(selectOptionSchema),
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
  dropout_prob: z.number().min(0).max(1),
});

const convLayerPreviewSchema = z.object({
  name: z.string(),
  in_channels: z.number().int().nonnegative(),
  out_channels: z.number().int().positive(),
  kernel_size: z.number().int().positive(),
  stride: z.number().int().positive(),
  padding: z.number().int().nonnegative(),
  input_height: z.number().int().positive(),
  input_width: z.number().int().positive(),
  output_height: z.number().int().positive(),
  output_width: z.number().int().positive(),
  dropped_height: z.number().int().nonnegative(),
  dropped_width: z.number().int().nonnegative(),
  params: z.number().int().nonnegative(),
});

const parameterGroupPreviewSchema = z.object({
  name: z.string(),
  params: z.number().int().nonnegative(),
});

const actionBranchPreviewSchema = z.object({
  name: z.string(),
  kind: z.string(),
  size: z.number().int().positive(),
  enabled: z.boolean(),
  mask_label: z.string().nullable().optional(),
});

const architectureNodePreviewSchema = z.object({
  id: z.string(),
  label: z.string(),
  detail: z.string(),
  params: z.number().int().nonnegative().nullable().optional(),
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
  action_branches: z.array(actionBranchPreviewSchema),
  continuous_action_dims: z.number().int().nonnegative(),
  discrete_action_logits: z.number().int().nonnegative(),
  parameter_groups: z.array(parameterGroupPreviewSchema),
  total_params: z.number().int().nonnegative(),
  architecture_lanes: z.array(architectureLanePreviewSchema),
});

export type ManagedRunConfig = z.infer<typeof managedRunConfigSchema>;
export type ManagedTemplate = z.infer<typeof managedTemplateSchema>;
export type ManagedDraft = z.infer<typeof managedDraftSchema>;
export type ManagedRun = z.infer<typeof managedRunSchema>;
export type ManagedRunMetricSample = z.infer<typeof managedRunMetricSampleSchema>;
export type TrackSamplingRuntimeEntry = z.infer<typeof trackSamplingRuntimeEntrySchema>;
export type TrackSamplingRuntimeState = z.infer<typeof trackSamplingRuntimeStateSchema>;
export type ConfigMetadata = z.infer<typeof configMetadataSchema>;
export type PolicyArchitecturePreview = z.infer<typeof policyArchitecturePreviewSchema>;
export type StateComponentConfig = z.infer<typeof stateComponentConfigSchema>;
export type StateFeatureDropoutConfig = z.infer<typeof stateFeatureDropoutConfigSchema>;
