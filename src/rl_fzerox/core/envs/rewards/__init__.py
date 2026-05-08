# src/rl_fzerox/core/envs/rewards/__init__.py
from rl_fzerox.core.envs.rewards.common import (
    RewardActionContext,
    RewardStep,
    RewardSummaryConfig,
    RewardTracker,
)
from rl_fzerox.core.envs.rewards.reward_main import RewardMainTracker, RewardMainWeights
from rl_fzerox.core.runtime_spec.schema import RewardConfig

CANONICAL_REWARD_NAME = "reward_main"
WeightT = RewardMainWeights


def build_reward_tracker(
    config: RewardConfig | None = None,
    *,
    max_episode_steps: int = 12_000,
) -> RewardTracker:
    """Construct one registered reward tracker by name."""

    resolved_config = config or RewardConfig()
    if resolved_config.name == CANONICAL_REWARD_NAME:
        return RewardMainTracker(
            weights=_weights_for(resolved_config),
            course_weights=_course_weights_for(resolved_config),
            max_episode_steps=max_episode_steps,
        )
    raise ValueError(f"Unsupported reward profile: {resolved_config.name!r}")


def _weights_for(config: RewardConfig) -> RewardMainWeights:
    """Map canonical reward schema fields to the reward-main weight dataclass."""

    return RewardMainWeights(
        energy_loss_epsilon=config.energy_loss_epsilon,
        progress_bucket_distance=config.progress_bucket_distance,
        progress_bucket_reward=config.progress_bucket_reward,
        progress_reward_interval_frames=config.progress_reward_interval_frames,
        suspend_progress_while_outside_track_bounds=(
            config.suspend_progress_while_outside_track_bounds
        ),
        outside_bounds_reentry_progress_distance_cap=(
            config.outside_bounds_reentry_progress_distance_cap
        ),
        outside_track_frame_penalty=config.outside_track_frame_penalty,
        time_penalty_per_frame=config.time_penalty_per_frame,
        reverse_time_penalty_scale=config.reverse_time_penalty_scale,
        slow_speed_time_penalty_scale=config.slow_speed_time_penalty_scale,
        slow_speed_time_penalty_start_kph=config.slow_speed_time_penalty_start_kph,
        slow_speed_time_penalty_power=config.slow_speed_time_penalty_power,
        lap_completion_bonus=config.lap_completion_bonus,
        lap_position_scale=config.lap_position_scale,
        damage_taken_frame_penalty=config.damage_taken_frame_penalty,
        damage_taken_streak_ramp_penalty=config.damage_taken_streak_ramp_penalty,
        damage_taken_streak_cap_frames=config.damage_taken_streak_cap_frames,
        manual_boost_reward=config.manual_boost_reward,
        boost_pad_reward=config.boost_pad_reward,
        boost_pad_reward_progress_window=config.boost_pad_reward_progress_window,
        energy_refill_progress_multiplier=config.energy_refill_progress_multiplier,
        dirt_progress_multiplier=config.dirt_progress_multiplier,
        ice_progress_multiplier=config.ice_progress_multiplier,
        dirt_entry_penalty=config.dirt_entry_penalty,
        ice_entry_penalty=config.ice_entry_penalty,
        energy_refill_collision_cooldown_frames=(
            config.energy_refill_collision_cooldown_frames
        ),
        air_brake_request_penalty=config.air_brake_request_penalty,
        lean_request_penalty=config.lean_request_penalty,
        airborne_pitch_up_penalty=config.airborne_pitch_up_penalty,
        grounded_pitch_penalty=config.grounded_pitch_penalty,
        grounded_pitch_deadzone=config.grounded_pitch_deadzone,
        airborne_landing_reward=config.airborne_landing_reward,
        collision_recoil_penalty=config.collision_recoil_penalty,
        failure_penalty=config.failure_penalty,
        truncation_penalty=config.truncation_penalty,
        step_reward_clip_min=config.step_reward_clip_min,
        step_reward_clip_max=config.step_reward_clip_max,
    )


def _course_weights_for(config: RewardConfig) -> dict[str, RewardMainWeights]:
    base = _weights_for(config)
    course_weights: dict[str, RewardMainWeights] = {}
    for raw_course_id, override in config.course_overrides.items():
        course_id = raw_course_id.strip()
        if not course_id:
            raise ValueError("reward course override keys must be non-empty course ids")
        course_weights[course_id] = RewardMainWeights(
            energy_loss_epsilon=(
                override.energy_loss_epsilon
                if override.energy_loss_epsilon is not None
                else base.energy_loss_epsilon
            ),
            progress_bucket_distance=(
                override.progress_bucket_distance
                if override.progress_bucket_distance is not None
                else base.progress_bucket_distance
            ),
            progress_bucket_reward=(
                override.progress_bucket_reward
                if override.progress_bucket_reward is not None
                else base.progress_bucket_reward
            ),
            progress_reward_interval_frames=(
                override.progress_reward_interval_frames
                if override.progress_reward_interval_frames is not None
                else base.progress_reward_interval_frames
            ),
            suspend_progress_while_outside_track_bounds=(
                override.suspend_progress_while_outside_track_bounds
                if override.suspend_progress_while_outside_track_bounds is not None
                else base.suspend_progress_while_outside_track_bounds
            ),
            outside_bounds_reentry_progress_distance_cap=(
                override.outside_bounds_reentry_progress_distance_cap
                if override.outside_bounds_reentry_progress_distance_cap is not None
                else base.outside_bounds_reentry_progress_distance_cap
            ),
            outside_track_frame_penalty=(
                override.outside_track_frame_penalty
                if override.outside_track_frame_penalty is not None
                else base.outside_track_frame_penalty
            ),
            time_penalty_per_frame=(
                override.time_penalty_per_frame
                if override.time_penalty_per_frame is not None
                else base.time_penalty_per_frame
            ),
            reverse_time_penalty_scale=(
                override.reverse_time_penalty_scale
                if override.reverse_time_penalty_scale is not None
                else base.reverse_time_penalty_scale
            ),
            slow_speed_time_penalty_scale=(
                override.slow_speed_time_penalty_scale
                if override.slow_speed_time_penalty_scale is not None
                else base.slow_speed_time_penalty_scale
            ),
            slow_speed_time_penalty_start_kph=(
                override.slow_speed_time_penalty_start_kph
                if override.slow_speed_time_penalty_start_kph is not None
                else base.slow_speed_time_penalty_start_kph
            ),
            slow_speed_time_penalty_power=(
                override.slow_speed_time_penalty_power
                if override.slow_speed_time_penalty_power is not None
                else base.slow_speed_time_penalty_power
            ),
            lap_completion_bonus=(
                override.lap_completion_bonus
                if override.lap_completion_bonus is not None
                else base.lap_completion_bonus
            ),
            lap_position_scale=(
                override.lap_position_scale
                if override.lap_position_scale is not None
                else base.lap_position_scale
            ),
            damage_taken_frame_penalty=(
                override.damage_taken_frame_penalty
                if override.damage_taken_frame_penalty is not None
                else base.damage_taken_frame_penalty
            ),
            damage_taken_streak_ramp_penalty=(
                override.damage_taken_streak_ramp_penalty
                if override.damage_taken_streak_ramp_penalty is not None
                else base.damage_taken_streak_ramp_penalty
            ),
            damage_taken_streak_cap_frames=(
                override.damage_taken_streak_cap_frames
                if override.damage_taken_streak_cap_frames is not None
                else base.damage_taken_streak_cap_frames
            ),
            manual_boost_reward=(
                override.manual_boost_reward
                if override.manual_boost_reward is not None
                else base.manual_boost_reward
            ),
            boost_pad_reward=(
                override.boost_pad_reward
                if override.boost_pad_reward is not None
                else base.boost_pad_reward
            ),
            boost_pad_reward_progress_window=(
                override.boost_pad_reward_progress_window
                if override.boost_pad_reward_progress_window is not None
                else base.boost_pad_reward_progress_window
            ),
            energy_refill_progress_multiplier=(
                override.energy_refill_progress_multiplier
                if override.energy_refill_progress_multiplier is not None
                else base.energy_refill_progress_multiplier
            ),
            dirt_progress_multiplier=(
                override.dirt_progress_multiplier
                if override.dirt_progress_multiplier is not None
                else base.dirt_progress_multiplier
            ),
            ice_progress_multiplier=(
                override.ice_progress_multiplier
                if override.ice_progress_multiplier is not None
                else base.ice_progress_multiplier
            ),
            dirt_entry_penalty=(
                override.dirt_entry_penalty
                if override.dirt_entry_penalty is not None
                else base.dirt_entry_penalty
            ),
            ice_entry_penalty=(
                override.ice_entry_penalty
                if override.ice_entry_penalty is not None
                else base.ice_entry_penalty
            ),
            energy_refill_collision_cooldown_frames=(
                override.energy_refill_collision_cooldown_frames
                if override.energy_refill_collision_cooldown_frames is not None
                else base.energy_refill_collision_cooldown_frames
            ),
            air_brake_request_penalty=(
                override.air_brake_request_penalty
                if override.air_brake_request_penalty is not None
                else base.air_brake_request_penalty
            ),
            lean_request_penalty=(
                override.lean_request_penalty
                if override.lean_request_penalty is not None
                else base.lean_request_penalty
            ),
            airborne_pitch_up_penalty=(
                override.airborne_pitch_up_penalty
                if override.airborne_pitch_up_penalty is not None
                else base.airborne_pitch_up_penalty
            ),
            grounded_pitch_penalty=(
                override.grounded_pitch_penalty
                if override.grounded_pitch_penalty is not None
                else base.grounded_pitch_penalty
            ),
            grounded_pitch_deadzone=(
                override.grounded_pitch_deadzone
                if override.grounded_pitch_deadzone is not None
                else base.grounded_pitch_deadzone
            ),
            airborne_landing_reward=(
                override.airborne_landing_reward
                if override.airborne_landing_reward is not None
                else base.airborne_landing_reward
            ),
            collision_recoil_penalty=(
                override.collision_recoil_penalty
                if override.collision_recoil_penalty is not None
                else base.collision_recoil_penalty
            ),
            failure_penalty=(
                override.failure_penalty
                if override.failure_penalty is not None
                else base.failure_penalty
            ),
            truncation_penalty=(
                override.truncation_penalty
                if override.truncation_penalty is not None
                else base.truncation_penalty
            ),
            step_reward_clip_min=(
                override.step_reward_clip_min
                if override.step_reward_clip_min is not None
                else base.step_reward_clip_min
            ),
            step_reward_clip_max=(
                override.step_reward_clip_max
                if override.step_reward_clip_max is not None
                else base.step_reward_clip_max
            ),
        )
    return course_weights


def reward_tracker_names() -> tuple[str, ...]:
    """Return the registered reward tracker names in insertion order."""

    return (CANONICAL_REWARD_NAME,)


__all__ = [
    "CANONICAL_REWARD_NAME",
    "RewardMainTracker",
    "RewardMainWeights",
    "RewardActionContext",
    "RewardStep",
    "RewardSummaryConfig",
    "RewardTracker",
    "build_reward_tracker",
    "reward_tracker_names",
]
