# src/rl_fzerox/core/envs/rewards/__init__.py
from rl_fzerox.core.config.schema import RewardConfig
from rl_fzerox.core.envs.rewards.common import (
    RewardActionContext,
    RewardStep,
    RewardSummaryConfig,
    RewardTracker,
)
from rl_fzerox.core.envs.rewards.race_v3 import RaceV3RewardTracker, RaceV3RewardWeights

DEFAULT_REWARD_NAME = "race_v3"


def build_reward_tracker(
    config: RewardConfig | None = None,
    *,
    max_episode_steps: int = 12_000,
) -> RewardTracker:
    """Construct one registered reward tracker by name."""

    resolved_config = config or RewardConfig()
    if resolved_config.name != DEFAULT_REWARD_NAME:
        raise ValueError(f"Unsupported reward profile: {resolved_config.name!r}")
    return RaceV3RewardTracker(
        weights=_race_v3_weights(resolved_config),
        max_episode_steps=max_episode_steps,
    )


def _race_v3_weights(config: RewardConfig) -> RaceV3RewardWeights:
    """Map schema fields to the race_v3 weights dataclass."""

    return RaceV3RewardWeights(
        energy_loss_epsilon=config.energy_loss_epsilon,
        progress_bucket_distance=config.progress_bucket_distance,
        progress_bucket_reward=config.progress_bucket_reward,
        progress_reward_interval_frames=config.progress_reward_interval_frames,
        defer_progress_reward_while_airborne=config.defer_progress_reward_while_airborne,
        airborne_progress_bucket_distance=config.airborne_progress_bucket_distance,
        airborne_progress_requires_nonascending=config.airborne_progress_requires_nonascending,
        airborne_progress_height_epsilon=config.airborne_progress_height_epsilon,
        time_penalty_per_frame=config.time_penalty_per_frame,
        reverse_time_penalty_scale=config.reverse_time_penalty_scale,
        low_speed_time_penalty_scale=config.low_speed_time_penalty_scale,
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
        energy_refill_collision_cooldown_frames=config.energy_refill_collision_cooldown_frames,
        energy_full_refill_lap_bonus=config.energy_full_refill_lap_bonus,
        energy_full_refill_min_gain_fraction=config.energy_full_refill_min_gain_fraction,
        gas_underuse_penalty=config.gas_underuse_penalty,
        gas_underuse_threshold=config.gas_underuse_threshold,
        steer_oscillation_penalty=config.steer_oscillation_penalty,
        steer_oscillation_deadzone=config.steer_oscillation_deadzone,
        steer_oscillation_cap=config.steer_oscillation_cap,
        steer_oscillation_power=config.steer_oscillation_power,
        lean_request_penalty=config.lean_request_penalty,
        lean_low_speed_penalty=config.lean_low_speed_penalty,
        lean_low_speed_penalty_max_speed_kph=config.lean_low_speed_penalty_max_speed_kph,
        airborne_landing_reward=config.airborne_landing_reward,
        collision_recoil_penalty=config.collision_recoil_penalty,
        failure_penalty=config.failure_penalty,
        truncation_penalty=config.truncation_penalty,
    )


def reward_tracker_names() -> tuple[str, ...]:
    """Return the registered reward tracker names in insertion order."""

    return (DEFAULT_REWARD_NAME,)


__all__ = [
    "DEFAULT_REWARD_NAME",
    "RaceV3RewardTracker",
    "RaceV3RewardWeights",
    "RewardActionContext",
    "RewardStep",
    "RewardSummaryConfig",
    "RewardTracker",
    "build_reward_tracker",
    "reward_tracker_names",
]
