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

RewardTrackerFactory = Callable[..., RewardTracker]
DEFAULT_REWARD_NAME = "race_v2"
REWARD_TRACKER_REGISTRY: dict[str, RewardTrackerFactory] = {
    DEFAULT_REWARD_NAME: RaceV2RewardTracker,
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
    weights = RaceV2RewardWeights(
        time_penalty_per_frame=resolved_config.time_penalty_per_frame,
        reverse_time_penalty_scale=resolved_config.reverse_time_penalty_scale,
        low_speed_time_penalty_scale=resolved_config.low_speed_time_penalty_scale,
        milestone_distance=resolved_config.milestone_distance,
        randomize_milestone_phase_on_reset=resolved_config.randomize_milestone_phase_on_reset,
        milestone_bonus=resolved_config.milestone_bonus,
        milestone_speed_scale=resolved_config.milestone_speed_scale,
        milestone_speed_bonus_cap=resolved_config.milestone_speed_bonus_cap,
        bootstrap_progress_scale=resolved_config.bootstrap_progress_scale,
        bootstrap_regress_penalty_scale=resolved_config.bootstrap_regress_penalty_scale,
        bootstrap_position_multiplier_scale=resolved_config.bootstrap_position_multiplier_scale,
        bootstrap_lap_count=resolved_config.bootstrap_lap_count,
        lap_1_completion_bonus=resolved_config.lap_1_completion_bonus,
        lap_2_completion_bonus=resolved_config.lap_2_completion_bonus,
        final_lap_completion_bonus=resolved_config.final_lap_completion_bonus,
        lap_position_scale=resolved_config.lap_position_scale,
        remaining_step_penalty_per_frame=resolved_config.remaining_step_penalty_per_frame,
        remaining_lap_penalty=resolved_config.remaining_lap_penalty,
        energy_loss_epsilon=resolved_config.energy_loss_epsilon,
        energy_loss_penalty_scale=resolved_config.energy_loss_penalty_scale,
        energy_loss_safe_fraction=resolved_config.energy_loss_safe_fraction,
        energy_loss_danger_power=resolved_config.energy_loss_danger_power,
        energy_gain_reward_scale=resolved_config.energy_gain_reward_scale,
        energy_gain_collision_cooldown_frames=resolved_config.energy_gain_collision_cooldown_frames,
        energy_full_refill_bonus=resolved_config.energy_full_refill_bonus,
        energy_full_refill_cooldown_frames=resolved_config.energy_full_refill_cooldown_frames,
        damage_taken_frame_penalty=resolved_config.damage_taken_frame_penalty,
        damage_taken_streak_ramp_penalty=resolved_config.damage_taken_streak_ramp_penalty,
        damage_taken_streak_cap_frames=resolved_config.damage_taken_streak_cap_frames,
        airborne_landing_reward=resolved_config.airborne_landing_reward,
        grounded_air_brake_penalty=resolved_config.grounded_air_brake_penalty,
        drive_axis_negative_penalty_scale=resolved_config.drive_axis_negative_penalty_scale,
        boost_pad_reward=resolved_config.boost_pad_reward,
        boost_pad_reward_cooldown_frames=resolved_config.boost_pad_reward_cooldown_frames,
        manual_boost_request_reward=resolved_config.manual_boost_request_reward,
        collision_recoil_penalty=resolved_config.collision_recoil_penalty,
        spinning_out_penalty=resolved_config.spinning_out_penalty,
        terminal_failure_base_penalty=resolved_config.terminal_failure_base_penalty,
        stuck_truncation_base_penalty=resolved_config.stuck_truncation_base_penalty,
        wrong_way_truncation_base_penalty=resolved_config.wrong_way_truncation_base_penalty,
        progress_stalled_truncation_base_penalty=(
            resolved_config.progress_stalled_truncation_base_penalty
        ),
        timeout_truncation_base_penalty=resolved_config.timeout_truncation_base_penalty,
        finish_position_scale=resolved_config.finish_position_scale,
    )
    return factory(weights=weights, max_episode_steps=max_episode_steps)


def reward_tracker_names() -> tuple[str, ...]:
    """Return the registered reward tracker names in insertion order."""

    return tuple(REWARD_TRACKER_REGISTRY)


__all__ = [
    "DEFAULT_REWARD_NAME",
    "RaceV2RewardTracker",
    "RaceV2RewardWeights",
    "REWARD_TRACKER_REGISTRY",
    "RewardActionContext",
    "RewardStep",
    "RewardSummaryConfig",
    "RewardTracker",
    "RewardTrackerFactory",
    "build_reward_tracker",
    "reward_tracker_names",
]
