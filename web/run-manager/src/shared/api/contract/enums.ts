// web/run-manager/src/shared/api/contract/enums.ts
import { z } from "zod";

export const runStatusSchema = z.enum([
  "created",
  "running",
  "paused",
  "stopped",
  "finished",
  "failed",
]);
export const runCommandSchema = z.enum(["pause", "stop"]);
export const observationPresetSchema = z.enum(["crop_72x96", "crop_84x84"]);
export const raceModeSchema = z.enum(["time_attack", "gp_race"]);
export const gpDifficultySchema = z.enum(["novice", "standard", "expert", "master"]);
export const trackSamplingModeSchema = z.enum([
  "equal",
  "step_balanced",
  // Hidden from new backend metadata, but accepted for legacy saved configs.
  "adaptive_step_balanced",
  "deficit_budget",
  "fixed_env",
]);
export const deficitBudgetDifficultyMetricSchema = z.enum([
  "completion_ema",
  "finish_ema",
  "mixed",
]);
export const vehicleSelectionModeSchema = z.enum(["fixed", "pool"]);
export const engineSettingModeSchema = z.enum(["fixed", "random_range", "adaptive_tuner"]);
export const engineTunerBackendSchema = z.enum(["bandit", "gaussian_process", "mlp_ensemble"]);
export const engineTunerObjectiveSchema = z.enum(["finish_time", "episode_return"]);
export const engineTuningSourceActionSchema = z.enum(["convert", "discard"]);
export const actionAxisModeSchema = z.enum(["continuous", "discrete"]);
export const actionDriveModeSchema = z.enum(["pwm", "on_off"]);
export const leanOutputModeSchema = z.enum([
  "three_way",
  "four_way_categorical",
  "independent_buttons",
]);
export const leanModeSchema = z.enum(["minimum_hold", "release_cooldown", "timer_assist", "raw"]);
export const rendererSchema = z.enum(["angrylion", "gliden64"]);
export const cameraSettingSchema = z.enum(["overhead", "close_behind", "regular", "wide"]);
export const watchDeviceSchema = z.enum(["cpu", "cuda"]);
export const stateComponentNameSchema = z.enum([
  "vehicle_state",
  "machine_context",
  "track_position",
  "surface_state",
  "course_context",
  "control_history",
]);
export const actionHistoryControlSchema = z.enum([
  "steer",
  "gas",
  "thrust",
  "air_brake",
  "boost",
  "lean",
  "pitch",
]);
export const convProfileSchema = z.enum(["nature", "impala_small", "impala_large", "custom"]);
export const customCnnActivationSchema = z.enum(["relu", "gelu"]);
export const auxiliaryStateTargetNameSchema = z.enum([
  "vehicle_state.speed_norm",
  "vehicle_state.energy_frac",
  "vehicle_state.reverse_active",
  "vehicle_state.airborne",
  "vehicle_state.boost_unlocked",
  "vehicle_state.boost_active",
  "vehicle_state.lateral_velocity_norm",
  "vehicle_state.sliding_active",
  "track_position.lap_progress",
  "track_position.edge_ratio",
  "track_position.height_above_ground_norm",
  "track_position.outside_track_bounds",
  "surface_state.on_refill_surface",
  "surface_state.on_dirt_surface",
  "surface_state.on_ice_surface",
  "course_context.builtin_course_id",
]);

export type WatchDevice = z.infer<typeof watchDeviceSchema>;
export type WatchRenderer = z.infer<typeof rendererSchema>;
export type EngineTunerBackend = z.infer<typeof engineTunerBackendSchema>;
export type EngineTunerObjective = z.infer<typeof engineTunerObjectiveSchema>;
export type EngineTuningSourceAction = z.infer<typeof engineTuningSourceActionSchema>;
