// web/run-manager/src/shared/api/contract/config.ts
import { z } from "zod";

import {
  actionAxisModeSchema,
  actionDriveModeSchema,
  actionHistoryControlSchema,
  auxiliaryStateTargetNameSchema,
  cameraSettingSchema,
  convProfileSchema,
  customCnnActivationSchema,
  engineSettingModeSchema,
  engineTunerBackendSchema,
  gpDifficultySchema,
  leanModeSchema,
  leanOutputModeSchema,
  observationPresetSchema,
  raceModeSchema,
  rendererSchema,
  stateComponentNameSchema,
  trackSamplingModeSchema,
  vehicleSelectionModeSchema,
} from "@/shared/api/contract/enums";
import {
  centeredEngineBuckets,
  ENGINE_SLIDER_STEP_MAX,
  enginePercentToSliderStep,
} from "@/shared/domain/engineBuckets";

export const customCnnLayerKindSchema = z.preprocess(
  (value) => (value === "residual" ? "residual_post" : value),
  z.enum(["conv", "residual_pre", "residual_post", "maxpool", "avgpool", "activation"]),
);

export const customConvLayerSchema = z
  .object({
    kind: customCnnLayerKindSchema.default("conv"),
    out_channels: z.number().int().positive(),
    kernel_size: z.number().int().positive(),
    stride: z.number().int().positive(),
    padding: z.number().int().nonnegative(),
    post_activation: z.boolean().default(true),
    activation: customCnnActivationSchema.nullable().optional(),
  })
  .refine(
    (layer) =>
      !isResidualCnnLayerKind(layer.kind) ||
      (layer.kernel_size % 2 === 1 && layer.padding === Math.floor(layer.kernel_size / 2)),
    {
      message: "residual CNN blocks require odd kernel_size and padding=kernel_size//2",
      path: ["padding"],
    },
  )
  .refine(
    (layer) =>
      layer.kind !== "activation" ||
      (layer.kernel_size === 1 && layer.stride === 1 && layer.padding === 0),
    {
      message: "activation CNN layers require kernel_size=1, stride=1, and padding=0",
      path: ["kind"],
    },
  );

function isResidualCnnLayerKind(kind: z.infer<typeof customCnnLayerKindSchema>) {
  return kind === "residual_pre" || kind === "residual_post";
}

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
  entropy_coefficients: z.record(z.string(), z.number().nonnegative()),
  actor_regularization: z.object({
    grounded_pitch_neutral_loss_weight: z.number().nonnegative(),
    pitch_std_cap_loss_weight: z.number().nonnegative(),
    grounded_pitch_std_cap: z.number().positive(),
    airborne_pitch_std_cap: z.number().positive(),
    steer_std_cap_loss_weight: z.number().nonnegative(),
    steer_std_cap: z.number().positive(),
    steer_signed_balance_loss_weight: z.number().nonnegative(),
    steer_signed_balance_deadzone: z.number().min(0).max(1),
    lean_signed_balance_loss_weight: z.number().nonnegative(),
    lean_signed_balance_deadzone: z.number().min(0).max(1),
  }),
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

type GpDifficulty = z.infer<typeof gpDifficultySchema>;
type RaceMode = z.infer<typeof raceModeSchema>;

function normalizedGpDifficulties({
  gpDifficulties,
  raceMode,
}: {
  gpDifficulties?: GpDifficulty[];
  raceMode: RaceMode;
}) {
  if (raceMode !== "gp_race") {
    return [];
  }
  const configured =
    gpDifficulties !== undefined && gpDifficulties.length > 0 ? gpDifficulties : ["novice"];
  return [...new Set(configured)];
}

const tracksConfigSchema = z
  .object({
    race_mode: raceModeSchema,
    gp_difficulties: z.array(gpDifficultySchema).optional(),
    include_x_cup: z.boolean(),
    x_cup_course_count: z.number().int().positive(),
    x_cup_auto_regeneration: z.object({
      enabled: z.boolean(),
      completion_threshold: z.number().min(0).max(1),
      min_episodes: z.number().int().positive(),
      max_episodes: z.number().int().positive().nullable(),
      ema_alpha: z.number().gt(0).max(1),
    }),
    sampling_mode: trackSamplingModeSchema,
    step_balance_update_episodes: z.number().int().positive(),
    step_balance_ema_alpha: z.number().gt(0).max(1),
    step_balance_max_weight_scale: z.number().min(1),
    adaptive_step_balance_completion_weight: z.number().nonnegative(),
    adaptive_step_balance_target_completion: z.number().min(0).max(1),
    adaptive_step_balance_min_confidence_episodes: z.number().int().positive(),
    adaptive_step_balance_confidence_scale: z.number().min(1),
    deficit_budget_uniform_fraction: z.number().min(0).max(1),
    deficit_budget_focus_sharpness: z.number().nonnegative(),
    deficit_budget_ema_alpha: z.number().gt(0).max(1),
    deficit_budget_weight_update_rollouts: z.number().int().positive(),
    selected_course_ids: z.array(z.string()),
  })
  .transform(({ gp_difficulties, ...tracks }) => ({
    ...tracks,
    gp_difficulties: normalizedGpDifficulties({
      gpDifficulties: gp_difficulties,
      raceMode: tracks.race_mode,
    }),
  }));

const vehicleConfigSchema = z
  .preprocess(
    (raw) => {
      if (raw === null || typeof raw !== "object" || Array.isArray(raw)) {
        return raw;
      }
      const data = { ...raw } as Record<string, unknown>;
      const legacySpacing = data.adaptive_engine_bandit_bucket_size;
      delete data.adaptive_engine_bandit_bucket_size;
      if (legacySpacing !== undefined && data.adaptive_engine_bandit_slider_spacing === undefined) {
        data.adaptive_engine_bandit_slider_spacing = legacySpacing;
      }
      return data;
    },
    z.object({
      selection_mode: vehicleSelectionModeSchema,
      selected_vehicle_ids: z.array(z.string()).min(1),
      engine_mode: engineSettingModeSchema,
      engine_setting_raw_value: z.number().int().min(0).max(ENGINE_SLIDER_STEP_MAX),
      engine_setting_min_raw_value: z.number().int().min(0).max(ENGINE_SLIDER_STEP_MAX),
      engine_setting_max_raw_value: z.number().int().min(0).max(ENGINE_SLIDER_STEP_MAX),
      adaptive_engine_tuner_backend: engineTunerBackendSchema.default("bandit"),
      adaptive_engine_bandit_slider_spacing: z
        .number()
        .int()
        .min(1)
        .max(ENGINE_SLIDER_STEP_MAX)
        .default(enginePercentToSliderStep(10)),
      adaptive_engine_stat_decay: z.number().gt(0).lt(1).default(0.995),
      adaptive_engine_ensemble_members: z.number().int().min(1).max(32).default(5),
      adaptive_engine_mlp_hidden_dim: z.number().int().min(4).max(512).default(32),
      adaptive_engine_mlp_training_steps: z.number().int().min(1).max(2048).default(48),
      adaptive_engine_mlp_learning_rate: z.number().gt(0).max(1).default(0.004),
      adaptive_engine_mlp_bootstrap_keep_probability: z.number().gt(0).max(1).default(0.8),
      adaptive_engine_mlp_warmup_successes: z.number().int().min(1).max(4096).default(32),
      adaptive_engine_uniform_exploration: z.number().min(0).max(1).default(0.05),
      adaptive_engine_greedy_plateau_seconds: z.number().min(0).max(30).default(1),
    }),
  )
  .refine(
    (vehicle) => vehicle.engine_setting_min_raw_value <= vehicle.engine_setting_max_raw_value,
    {
      message: "engine_setting_min_raw_value must be <= engine_setting_max_raw_value",
      path: ["engine_setting_min_raw_value"],
    },
  )
  .refine(
    (vehicle) =>
      vehicle.engine_mode !== "adaptive_tuner" ||
      vehicle.adaptive_engine_tuner_backend !== "bandit" ||
      centeredEngineBuckets({
        sliderSpacing: vehicle.adaptive_engine_bandit_slider_spacing,
        minimum: vehicle.engine_setting_min_raw_value,
        maximum: vehicle.engine_setting_max_raw_value,
      }).length > 0,
    {
      message: "engine slider spacing has no values inside the engine range",
      path: ["adaptive_engine_bandit_slider_spacing"],
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
    air_brake_episode_mask_probability: z.number().min(0).max(1).default(0),
    air_brake_pulse_frames: z.number().int().nonnegative().default(0),
    continuous_air_brake_deadzone: z.number().min(0).lt(1),
    continuous_air_brake_full_threshold: z.number().gt(0).max(1),
    continuous_air_brake_min_duty: z.number().min(0).max(1),
    include_boost: z.boolean(),
    enable_boost: z.boolean(),
    mask_boost_when_active: z.boolean(),
    mask_boost_when_airborne: z.boolean(),
    boost_decision_interval_steps: z.number().int().positive(),
    boost_request_lockout_frames: z.number().int().nonnegative(),
    boost_unmask_max_speed_kph: z.number().nonnegative().nullable(),
    boost_min_energy_fraction: z.number().min(0).max(1),
    include_lean: z.boolean(),
    enable_lean: z.boolean(),
    lean_output_mode: leanOutputModeSchema,
    lean_mode: leanModeSchema,
    lean_unmask_min_speed_kph: z.number().nonnegative().nullable(),
    lean_initial_lockout_frames: z.number().int().nonnegative(),
    lean_episode_mask_probability: z.number().min(0).max(1).default(0),
    include_spin: z.boolean().default(false),
    enable_spin: z.boolean().default(false),
    spin_cooldown_frames: z.number().int().nonnegative(),
    spin_episode_mask_probability: z.number().min(0).max(1).default(0),
    include_pitch: z.boolean(),
    enable_pitch: z.boolean(),
    pitch_mode: actionAxisModeSchema,
    mask_pitch_on_ground: z.boolean(),
    pitch_deadzone: z.number().min(0).lt(1),
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

const environmentConfigSchema = z
  .object({
    max_episode_steps: z.number().int().positive(),
    progress_frontier_stall_limit_frames: z.number().int().positive().nullable(),
    progress_frontier_epsilon: z.number().nonnegative(),
    renderer: rendererSchema,
    camera_setting: cameraSettingSchema,
    randomize_gp_lives_on_reset: z.boolean().default(false),
    gp_lives_jitter_min: z.number().int().default(0),
    gp_lives_jitter_max: z.number().int().default(4),
  })
  .refine((environment) => environment.gp_lives_jitter_min <= environment.gp_lives_jitter_max, {
    message: "gp_lives_jitter_min must be <= gp_lives_jitter_max",
    path: ["gp_lives_jitter_min"],
  });

export const stateComponentConfigSchema = z.object({
  name: stateComponentNameSchema,
  encoding: z.enum(["none", "one_hot_builtin"]).nullable(),
  progress_source: z.enum(["lap_progress", "segment_progress", "none"]).nullable(),
  length: z.number().int().positive().max(16).nullable(),
  controls: z.array(actionHistoryControlSchema).nullable(),
  included_features: z.array(z.string()).nullable(),
});

export const stateFeatureDropoutConfigSchema = z.object({
  name: z.string(),
  dropout_prob: z.number().min(0).max(1),
});

const observationCustomResolutionSchema = z.object({
  mode: z.literal("custom"),
  height: z.number().int().min(32).max(208),
  width: z.number().int().min(32).max(592),
});

const observationResolutionSchema = z.discriminatedUnion("mode", [
  z.object({
    mode: z.literal("preset"),
    preset: observationPresetSchema,
  }),
  observationCustomResolutionSchema,
  z.object({
    mode: z.literal("source_crop"),
  }),
]);

const observationConfigSchema = z.object({
  resolution: observationResolutionSchema,
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
  image_projection_activation: z.enum(["relu", "gelu", "tanh"]).default("relu"),
  state_net_arch: z.array(z.number().int().positive()),
  state_activation: z.enum(["relu", "gelu", "tanh"]).default("relu"),
  fusion_features_dim: z.number().int().positive().nullable(),
  fusion_activation: z.enum(["relu", "gelu", "tanh"]).default("relu"),
  layer_norm: z.boolean(),
  layer_norm_activation: z.enum(["relu", "gelu", "tanh"]).nullable().default(null),
  activation: z.enum(["relu", "gelu", "tanh"]).default("relu"),
  recurrent_enabled: z.boolean(),
  recurrent_hidden_size: z.number().int().positive(),
  recurrent_n_lstm_layers: z.number().int().positive(),
  recurrent_shared_lstm: z.boolean(),
  recurrent_enable_critic_lstm: z.boolean(),
  pi_net_arch: z.array(z.number().int().positive()),
  vf_net_arch: z.array(z.number().int().positive()),
  gas_on_logit: z.number(),
  air_brake_on_logit: z.number().default(0),
  spin_idle_logit: z.number().default(0),
  auxiliary_state_enabled: z.boolean(),
  auxiliary_state_head_arch: z.array(z.number().int().positive()),
  auxiliary_state_losses: z.array(
    z.object({
      name: auxiliaryStateTargetNameSchema,
      weight: z.number().positive(),
      grounded_only: z.boolean(),
    }),
  ),
});

const rewardConfigSchema = z
  .object({
    time_penalty_per_frame: z.number(),
    progress_bucket_distance: z.number().nonnegative(),
    progress_bucket_reward: z.number().nonnegative(),
    progress_reward_interval_frames: z.number().int().positive(),
    suspend_progress_while_outside_track_bounds: z.boolean(),
    progress_track_distance_tolerance: z.number().nonnegative(),
    progress_speed_min_kph: z.number().nonnegative(),
    progress_speed_min_multiplier: z.number().nonnegative(),
    progress_speed_reference_kph: z.number().positive(),
    progress_speed_max_kph: z.number().positive(),
    progress_speed_max_multiplier: z.number().nonnegative(),
    progress_speed_curve_power: z.number().positive(),
    position_progress_min_multiplier: z.number().nonnegative(),
    position_progress_max_multiplier: z.number().nonnegative(),
    outside_track_recovery_reward: z.number().nonnegative(),
    outside_track_recovery_reward_cap: z.number().nonnegative(),
    outside_track_recovery_airborne_grace_frames: z.number().int().nonnegative(),
    lap_completion_bonus: z.number().nonnegative(),
    lap_position_scale: z.number().nonnegative(),
    ko_star_reward: z.number().nonnegative(),
    energy_loss_epsilon: z.number().nonnegative(),
    energy_refill_progress_multiplier: z.number().min(1),
    dirt_progress_multiplier: z.number().nonnegative(),
    ice_progress_multiplier: z.number().nonnegative(),
    dirt_entry_penalty: z.number().max(0),
    ice_entry_penalty: z.number().max(0),
    energy_refill_collision_cooldown_frames: z.number().int().nonnegative(),
    air_brake_request_penalty: z.number().max(0),
    spin_request_penalty: z.number().max(0),
    manual_boost_reward: z.number().nonnegative(),
    manual_boost_reward_energy_shaping: z.boolean(),
    manual_boost_reward_min_energy_fraction: z.number().min(0).lt(1),
    manual_boost_reward_min_energy_value: z.number(),
    manual_boost_reward_full_energy_fraction: z.number().positive().max(1),
    manual_boost_reward_energy_curve: z.enum(["linear", "smoothstep"]),
    boost_pad_reward_before_unlock: z.number().nonnegative(),
    boost_pad_reward_after_unlock: z.number().nonnegative(),
    boost_pad_reward_progress_window: z.number().positive(),
    lean_request_penalty: z.number().max(0),
    lean_activation_penalty: z.number().max(0),
    grounded_pitch_penalty: z.number().max(0),
    impact_frame_penalty: z.number().max(0),
    energy_loss_penalty: z.number().max(0),
    energy_gain_reward: z.number().nonnegative(),
    airborne_landing_reward: z.number(),
    airborne_landing_grace_frames: z.number().int().nonnegative(),
    airborne_landing_min_peak_height: z.number().nonnegative(),
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
  )
  .refine((reward) => reward.progress_speed_reference_kph > reward.progress_speed_min_kph, {
    message: "progress_speed_reference_kph must be greater than min kph",
    path: ["progress_speed_reference_kph"],
  })
  .refine((reward) => reward.progress_speed_max_kph > reward.progress_speed_reference_kph, {
    message: "progress_speed_max_kph must be greater than reference kph",
    path: ["progress_speed_max_kph"],
  })
  .refine(
    (reward) => reward.position_progress_min_multiplier <= reward.position_progress_max_multiplier,
    {
      message: "position_progress_min_multiplier must be <= position_progress_max_multiplier",
      path: ["position_progress_min_multiplier"],
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

export type ManagedRunConfig = z.infer<typeof managedRunConfigSchema>;
export type StateComponentConfig = z.infer<typeof stateComponentConfigSchema>;
export type StateFeatureDropoutConfig = z.infer<typeof stateFeatureDropoutConfigSchema>;
