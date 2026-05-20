// src/rl_fzerox/apps/run_manager/web/src/features/configurator/sections/rewardDefaults.ts
import type { ManagedRunConfig } from "@/shared/api/contract";

type RewardConfig = ManagedRunConfig["reward"];
type RewardPatch = Partial<RewardConfig>;

export function timePressureDefaults(reward: RewardConfig): RewardPatch {
  return {
    reverse_time_penalty_scale: reward.reverse_time_penalty_scale,
    slow_speed_time_penalty_power: reward.slow_speed_time_penalty_power,
    slow_speed_time_penalty_scale: reward.slow_speed_time_penalty_scale,
    slow_speed_time_penalty_start_kph: reward.slow_speed_time_penalty_start_kph,
    time_penalty_per_frame: reward.time_penalty_per_frame,
  };
}

export function progressDefaults(reward: RewardConfig): RewardPatch {
  return {
    outside_bounds_reentry_progress_distance_cap:
      reward.outside_bounds_reentry_progress_distance_cap,
    progress_bucket_distance: reward.progress_bucket_distance,
    progress_bucket_reward: reward.progress_bucket_reward,
    progress_reward_interval_frames: reward.progress_reward_interval_frames,
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
    lean_request_penalty: reward.lean_request_penalty,
    manual_boost_reward: reward.manual_boost_reward,
  };
}

export function damageDefaults(reward: RewardConfig): RewardPatch {
  return {
    collision_recoil_penalty: reward.collision_recoil_penalty,
    damage_taken_frame_penalty: reward.damage_taken_frame_penalty,
    damage_taken_streak_cap_frames: reward.damage_taken_streak_cap_frames,
    damage_taken_streak_ramp_penalty: reward.damage_taken_streak_ramp_penalty,
    energy_loss_epsilon: reward.energy_loss_epsilon,
    failure_penalty: reward.failure_penalty,
    step_reward_clip_max: reward.step_reward_clip_max,
    step_reward_clip_min: reward.step_reward_clip_min,
    truncation_penalty: reward.truncation_penalty,
  };
}
