// src/rl_fzerox/apps/run_manager/web/src/features/configurator/sections/rewardDefaults.ts
import type { ManagedRunConfig } from "@/shared/api/contract";

type RewardConfig = ManagedRunConfig["reward"];
type RewardPatch = Partial<RewardConfig>;

export function timePressureDefaults(reward: RewardConfig): RewardPatch {
  return {
    time_penalty_per_frame: reward.time_penalty_per_frame,
  };
}

export function progressDefaults(reward: RewardConfig): RewardPatch {
  return {
    progress_bucket_distance: reward.progress_bucket_distance,
    progress_bucket_reward: reward.progress_bucket_reward,
    progress_reward_interval_frames: reward.progress_reward_interval_frames,
    progress_speed_curve_power: reward.progress_speed_curve_power,
    progress_speed_max_kph: reward.progress_speed_max_kph,
    progress_speed_max_multiplier: reward.progress_speed_max_multiplier,
    progress_speed_min_kph: reward.progress_speed_min_kph,
    progress_speed_min_multiplier: reward.progress_speed_min_multiplier,
    progress_speed_reference_kph: reward.progress_speed_reference_kph,
    progress_track_distance_tolerance: reward.progress_track_distance_tolerance,
    suspend_progress_while_outside_track_bounds: reward.suspend_progress_while_outside_track_bounds,
  };
}

export function boundsDefaults(reward: RewardConfig): RewardPatch {
  return {
    airborne_landing_reward: reward.airborne_landing_reward,
    airborne_landing_grace_frames: reward.airborne_landing_grace_frames,
    airborne_landing_min_peak_height: reward.airborne_landing_min_peak_height,
    outside_track_recovery_airborne_grace_frames:
      reward.outside_track_recovery_airborne_grace_frames,
    outside_track_recovery_reward_cap: reward.outside_track_recovery_reward_cap,
    outside_track_recovery_reward: reward.outside_track_recovery_reward,
  };
}

export function trackEventDefaults(reward: RewardConfig): RewardPatch {
  return {
    dirt_entry_penalty: reward.dirt_entry_penalty,
    dirt_progress_multiplier: reward.dirt_progress_multiplier,
    ice_entry_penalty: reward.ice_entry_penalty,
    ice_progress_multiplier: reward.ice_progress_multiplier,
    ko_star_reward: reward.ko_star_reward,
    lap_completion_bonus: reward.lap_completion_bonus,
    lap_position_scale: reward.lap_position_scale,
    position_progress_max_multiplier: reward.position_progress_max_multiplier,
    position_progress_min_multiplier: reward.position_progress_min_multiplier,
  };
}

export function energyDefaults(reward: RewardConfig): RewardPatch {
  return {
    energy_refill_collision_cooldown_frames: reward.energy_refill_collision_cooldown_frames,
    energy_refill_progress_multiplier: reward.energy_refill_progress_multiplier,
  };
}

export function actionDefaults(reward: RewardConfig): RewardPatch {
  return {
    air_brake_request_penalty: reward.air_brake_request_penalty,
    boost_pad_reward: reward.boost_pad_reward,
    boost_pad_reward_progress_window: reward.boost_pad_reward_progress_window,
    grounded_pitch_penalty: reward.grounded_pitch_penalty,
    lean_activation_penalty: reward.lean_activation_penalty,
    lean_request_penalty: reward.lean_request_penalty,
    manual_boost_reward: reward.manual_boost_reward,
    manual_boost_reward_energy_curve: reward.manual_boost_reward_energy_curve,
    manual_boost_reward_energy_shaping: reward.manual_boost_reward_energy_shaping,
    manual_boost_reward_full_energy_fraction: reward.manual_boost_reward_full_energy_fraction,
    manual_boost_reward_min_energy_fraction: reward.manual_boost_reward_min_energy_fraction,
    manual_boost_reward_min_energy_value: reward.manual_boost_reward_min_energy_value,
    spin_request_penalty: reward.spin_request_penalty,
  };
}

export function damageDefaults(reward: RewardConfig): RewardPatch {
  return {
    energy_gain_reward: reward.energy_gain_reward,
    energy_loss_penalty: reward.energy_loss_penalty,
    energy_loss_epsilon: reward.energy_loss_epsilon,
    failure_penalty: reward.failure_penalty,
    impact_frame_penalty: reward.impact_frame_penalty,
    step_reward_clip_max: reward.step_reward_clip_max,
    step_reward_clip_min: reward.step_reward_clip_min,
    truncation_penalty: reward.truncation_penalty,
  };
}
