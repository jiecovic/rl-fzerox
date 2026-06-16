// web/run-manager/src/entities/runConfig/ui/sections/action/branches/model.ts
import type { AuxiliaryActionConfig } from "@/entities/runConfig/ui/sections/action/branches/types";

export function resetAuxiliaryBranchesAction({
  action,
  checkpointLocked,
  defaultAction,
}: {
  action: AuxiliaryActionConfig;
  checkpointLocked: boolean;
  defaultAction: AuxiliaryActionConfig;
}) {
  return {
    include_air_brake: checkpointLocked
      ? action.include_air_brake
      : defaultAction.include_air_brake,
    air_brake_mode: checkpointLocked ? action.air_brake_mode : defaultAction.air_brake_mode,
    enable_air_brake: defaultAction.enable_air_brake,
    mask_air_brake_on_ground: defaultAction.mask_air_brake_on_ground,
    air_brake_episode_mask_probability: defaultAction.air_brake_episode_mask_probability,
    continuous_air_brake_deadzone: defaultAction.continuous_air_brake_deadzone,
    continuous_air_brake_full_threshold: defaultAction.continuous_air_brake_full_threshold,
    continuous_air_brake_min_duty: defaultAction.continuous_air_brake_min_duty,
    include_boost: checkpointLocked ? action.include_boost : defaultAction.include_boost,
    enable_boost: defaultAction.enable_boost,
    mask_boost_when_active: defaultAction.mask_boost_when_active,
    mask_boost_when_airborne: defaultAction.mask_boost_when_airborne,
    boost_decision_interval_steps: defaultAction.boost_decision_interval_steps,
    boost_request_lockout_frames: defaultAction.boost_request_lockout_frames,
    boost_unmask_max_speed_kph: defaultAction.boost_unmask_max_speed_kph,
    boost_min_energy_fraction: defaultAction.boost_min_energy_fraction,
    include_lean: checkpointLocked ? action.include_lean : defaultAction.include_lean,
    enable_lean: defaultAction.enable_lean,
    lean_output_mode: checkpointLocked ? action.lean_output_mode : defaultAction.lean_output_mode,
    lean_mode: defaultAction.lean_mode,
    lean_unmask_min_speed_kph: defaultAction.lean_unmask_min_speed_kph,
    lean_initial_lockout_frames: defaultAction.lean_initial_lockout_frames,
    lean_episode_mask_probability: defaultAction.lean_episode_mask_probability,
    include_spin: checkpointLocked ? action.include_spin : defaultAction.include_spin,
    enable_spin: checkpointLocked && action.include_spin ? true : defaultAction.enable_spin,
    spin_cooldown_frames: defaultAction.spin_cooldown_frames,
    spin_episode_mask_probability: defaultAction.spin_episode_mask_probability,
    include_pitch: checkpointLocked ? action.include_pitch : defaultAction.include_pitch,
    enable_pitch: defaultAction.enable_pitch,
    pitch_mode: checkpointLocked ? action.pitch_mode : defaultAction.pitch_mode,
    mask_pitch_on_ground: defaultAction.mask_pitch_on_ground,
    pitch_buckets: checkpointLocked ? action.pitch_buckets : defaultAction.pitch_buckets,
  } satisfies Partial<AuxiliaryActionConfig>;
}
