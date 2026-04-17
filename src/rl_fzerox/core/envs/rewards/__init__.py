# src/rl_fzerox/core/envs/rewards/__init__.py
from collections.abc import Callable

from rl_fzerox.core.config.schema import RewardConfig
from rl_fzerox.core.envs.rewards.common import (
    RewardActionContext,
    RewardStep,
    RewardSummaryConfig,
    RewardTracker,
)
from rl_fzerox.core.envs.rewards.race_v2 import RaceV2RewardTracker, RaceV2RewardWeights
from rl_fzerox.core.envs.rewards.race_v3 import RaceV3RewardTracker, RaceV3RewardWeights
from rl_fzerox.core.envs.rewards.race_v4 import RaceV4RewardTracker, RaceV4RewardWeights

RewardTrackerFactory = Callable[[RewardConfig, int], RewardTracker]
DEFAULT_REWARD_NAME = "race_v2"
REWARD_TRACKER_REGISTRY: dict[str, RewardTrackerFactory] = {
    DEFAULT_REWARD_NAME: lambda config, max_episode_steps: RaceV2RewardTracker(
        weights=_race_v2_weights(config),
        max_episode_steps=max_episode_steps,
    ),
    "race_v3": lambda config, max_episode_steps: RaceV3RewardTracker(
        weights=_race_v3_weights(config),
        max_episode_steps=max_episode_steps,
    ),
    "race_v4": lambda config, max_episode_steps: RaceV4RewardTracker(
        weights=_race_v4_weights(config),
        max_episode_steps=max_episode_steps,
    ),
}


def build_reward_tracker(
    config: RewardConfig | None = None,
    *,
    max_episode_steps: int = 12_000,
) -> RewardTracker:
    """Construct one registered reward tracker by name."""

    resolved_config = config or RewardConfig()
    factory = REWARD_TRACKER_REGISTRY.get(resolved_config.name)
    if factory is None:
        raise ValueError(f"Unsupported reward profile: {resolved_config.name!r}")
    return factory(resolved_config, max_episode_steps)


def _race_v2_weights(config: RewardConfig) -> RaceV2RewardWeights:
    """Map schema fields to the race_v2 weights dataclass."""

    weights = RaceV2RewardWeights(
        time_penalty_per_frame=config.time_penalty_per_frame,
        reverse_time_penalty_scale=config.reverse_time_penalty_scale,
        low_speed_time_penalty_scale=config.low_speed_time_penalty_scale,
        milestone_distance=config.milestone_distance,
        randomize_milestone_phase_on_reset=config.randomize_milestone_phase_on_reset,
        milestone_bonus=config.milestone_bonus,
        milestone_speed_scale=config.milestone_speed_scale,
        milestone_speed_bonus_cap=config.milestone_speed_bonus_cap,
        bootstrap_progress_scale=config.bootstrap_progress_scale,
        bootstrap_regress_penalty_scale=config.bootstrap_regress_penalty_scale,
        progress_reward_interval_frames=config.progress_reward_interval_frames,
        bootstrap_position_multiplier_scale=config.bootstrap_position_multiplier_scale,
        bootstrap_lap_count=config.bootstrap_lap_count,
        lap_1_completion_bonus=config.lap_1_completion_bonus,
        lap_2_completion_bonus=config.lap_2_completion_bonus,
        final_lap_completion_bonus=config.final_lap_completion_bonus,
        lap_position_scale=config.lap_position_scale,
        remaining_step_penalty_per_frame=config.remaining_step_penalty_per_frame,
        remaining_lap_penalty=config.remaining_lap_penalty,
        energy_loss_epsilon=config.energy_loss_epsilon,
        energy_loss_penalty_scale=config.energy_loss_penalty_scale,
        energy_loss_safe_fraction=config.energy_loss_safe_fraction,
        energy_loss_danger_power=config.energy_loss_danger_power,
        energy_gain_reward_scale=config.energy_gain_reward_scale,
        energy_gain_collision_cooldown_frames=config.energy_gain_collision_cooldown_frames,
        energy_full_refill_bonus=config.energy_full_refill_bonus,
        energy_full_refill_cooldown_frames=config.energy_full_refill_cooldown_frames,
        damage_taken_frame_penalty=config.damage_taken_frame_penalty,
        damage_taken_streak_ramp_penalty=config.damage_taken_streak_ramp_penalty,
        damage_taken_streak_cap_frames=config.damage_taken_streak_cap_frames,
        airborne_landing_reward=config.airborne_landing_reward,
        grounded_air_brake_penalty=config.grounded_air_brake_penalty,
        drive_axis_negative_penalty_scale=config.drive_axis_negative_penalty_scale,
        boost_pad_reward=config.boost_pad_reward,
        boost_pad_reward_cooldown_frames=config.boost_pad_reward_cooldown_frames,
        manual_boost_request_reward=config.manual_boost_request_reward,
        collision_recoil_penalty=config.collision_recoil_penalty,
        spinning_out_penalty=config.spinning_out_penalty,
        terminal_failure_base_penalty=config.terminal_failure_base_penalty,
        stuck_truncation_base_penalty=config.stuck_truncation_base_penalty,
        wrong_way_truncation_base_penalty=config.wrong_way_truncation_base_penalty,
        progress_stalled_truncation_base_penalty=config.progress_stalled_truncation_base_penalty,
        timeout_truncation_base_penalty=config.timeout_truncation_base_penalty,
        finish_position_scale=config.finish_position_scale,
    )
    return weights


def _race_v3_weights(config: RewardConfig) -> RaceV3RewardWeights:
    """Map schema fields to the race_v3 weights dataclass."""

    return RaceV3RewardWeights(
        energy_loss_epsilon=config.energy_loss_epsilon,
        progress_bucket_distance=config.progress_bucket_distance,
        progress_bucket_reward=config.progress_bucket_reward,
        progress_reward_interval_frames=config.progress_reward_interval_frames,
        time_penalty_per_frame=config.time_penalty_per_frame,
        reverse_time_penalty_scale=config.reverse_time_penalty_scale,
        low_speed_time_penalty_scale=config.low_speed_time_penalty_scale,
        lap_completion_bonus=config.lap_completion_bonus,
        lap_position_scale=config.lap_position_scale,
        damage_taken_frame_penalty=config.damage_taken_frame_penalty,
        damage_taken_streak_ramp_penalty=config.damage_taken_streak_ramp_penalty,
        damage_taken_streak_cap_frames=config.damage_taken_streak_cap_frames,
        boost_pad_reward=config.boost_pad_reward,
        boost_pad_reward_progress_window=config.boost_pad_reward_progress_window,
        energy_gain_reward_scale=config.energy_gain_reward_scale,
        energy_gain_collision_cooldown_frames=config.energy_gain_collision_cooldown_frames,
        energy_full_refill_lap_bonus=config.energy_full_refill_lap_bonus,
        airborne_landing_reward=config.airborne_landing_reward,
        collision_recoil_penalty=config.collision_recoil_penalty,
        failure_penalty=config.failure_penalty,
        truncation_penalty=config.truncation_penalty,
    )


def _race_v4_weights(config: RewardConfig) -> RaceV4RewardWeights:
    """Map schema fields to the race_v4 weights dataclass."""

    return RaceV4RewardWeights(
        energy_loss_epsilon=config.energy_loss_epsilon,
        time_penalty_per_frame=config.time_penalty_per_frame,
        reverse_time_penalty_scale=config.reverse_time_penalty_scale,
        low_speed_time_penalty_scale=config.low_speed_time_penalty_scale,
        milestone_distance=config.milestone_distance,
        randomize_milestone_phase_on_reset=config.randomize_milestone_phase_on_reset,
        milestone_bonus=config.milestone_bonus,
        lap_1_completion_bonus=config.lap_1_completion_bonus,
        lap_2_completion_bonus=config.lap_2_completion_bonus,
        final_lap_completion_bonus=config.final_lap_completion_bonus,
        lap_position_scale=config.lap_position_scale,
        remaining_step_penalty_per_frame=config.remaining_step_penalty_per_frame,
        remaining_lap_penalty=config.remaining_lap_penalty,
        damage_taken_frame_penalty=config.damage_taken_frame_penalty,
        damage_taken_streak_ramp_penalty=config.damage_taken_streak_ramp_penalty,
        damage_taken_streak_cap_frames=config.damage_taken_streak_cap_frames,
        terminal_failure_base_penalty=config.terminal_failure_base_penalty,
        stuck_truncation_base_penalty=config.stuck_truncation_base_penalty,
        wrong_way_truncation_base_penalty=config.wrong_way_truncation_base_penalty,
        progress_stalled_truncation_base_penalty=config.progress_stalled_truncation_base_penalty,
        timeout_truncation_base_penalty=config.timeout_truncation_base_penalty,
        finish_position_scale=config.finish_position_scale,
    )


def reward_tracker_names() -> tuple[str, ...]:
    """Return the registered reward tracker names in insertion order."""

    return tuple(REWARD_TRACKER_REGISTRY)


__all__ = [
    "DEFAULT_REWARD_NAME",
    "RaceV2RewardTracker",
    "RaceV2RewardWeights",
    "RaceV3RewardTracker",
    "RaceV3RewardWeights",
    "RaceV4RewardTracker",
    "RaceV4RewardWeights",
    "REWARD_TRACKER_REGISTRY",
    "RewardActionContext",
    "RewardStep",
    "RewardSummaryConfig",
    "RewardTracker",
    "RewardTrackerFactory",
    "build_reward_tracker",
    "reward_tracker_names",
]
