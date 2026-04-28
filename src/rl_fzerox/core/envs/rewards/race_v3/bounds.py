# src/rl_fzerox/core/envs/rewards/race_v3/bounds.py
from __future__ import annotations

from fzerox_emulator import FZeroXTelemetry, StepSummary
from rl_fzerox.core.envs.rewards.race_v3.progress import FrontierReward
from rl_fzerox.core.envs.rewards.race_v3.weights import RaceV3RewardWeights
from rl_fzerox.core.envs.track_bounds import track_edge_state


def cap_outside_bounds_reentry_reward(
    frontier_reward: FrontierReward,
    *,
    weights: RaceV3RewardWeights,
) -> FrontierReward:
    """Cap deferred re-entry progress reward without moving the frontier backward."""

    distance_cap = weights.outside_bounds_reentry_progress_distance_cap
    if distance_cap is None:
        return frontier_reward
    reward_cap = (
        max(float(distance_cap), 0.0)
        / weights.progress_bucket_distance
        * weights.progress_bucket_reward
    )
    if frontier_reward.progress <= reward_cap:
        return frontier_reward
    return FrontierReward(
        progress=reward_cap,
        ground_effect_adjustment=0.0,
        energy_refill_bonus=0.0,
    )


def airborne_offtrack_excess(telemetry: FZeroXTelemetry | None) -> float | None:
    """Return lateral edge excess while airborne and outside bounds."""

    if telemetry is None or not telemetry.player.airborne:
        return None
    edge_ratio = track_edge_state(telemetry.player).ratio
    if edge_ratio is None:
        return None
    outside_excess = max(abs(edge_ratio) - 1.0, 0.0)
    if outside_excess <= 0.0:
        return None
    return outside_excess


def airborne_height_above_ground(telemetry: FZeroXTelemetry | None) -> float | None:
    """Return airborne height, or None when not airborne."""

    if telemetry is None or not telemetry.player.airborne:
        return None
    return float(telemetry.player.height_above_ground)


def airborne_descending(
    *,
    previous_height: float | None,
    telemetry: FZeroXTelemetry,
    weights: RaceV3RewardWeights,
) -> bool:
    """Return whether the player is descending enough to reward off-track recovery."""

    current_height = airborne_height_above_ground(telemetry)
    if previous_height is None or current_height is None:
        return False
    epsilon = float(weights.airborne_offtrack_recovery_descend_epsilon)
    return current_height < previous_height - epsilon


def airborne_offtrack_penalty(
    summary: StepSummary,
    outside_excess: float | None,
    *,
    weights: RaceV3RewardWeights,
) -> float:
    """Penalize airborne distance beyond the lateral track edge."""

    scale = weights.airborne_offtrack_penalty_scale
    if scale <= 0.0 or outside_excess is None:
        return 0.0
    return -scale * (outside_excess**2) * max(int(summary.frames_run), 1)


def airborne_offtrack_recovery_reward(
    *,
    previous_excess: float | None,
    current_excess: float | None,
    descending: bool,
    weights: RaceV3RewardWeights,
) -> float:
    """Shape airborne off-track recovery as a non-farmable potential delta."""

    scale = weights.airborne_offtrack_recovery_reward_scale
    if scale <= 0.0 or current_excess is None:
        return 0.0
    reward = scale * ((previous_excess or 0.0) - current_excess)
    if (
        reward > 0.0
        and weights.airborne_offtrack_recovery_requires_descending
        and not descending
    ):
        return 0.0
    return reward
