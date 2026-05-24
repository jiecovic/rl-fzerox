# src/rl_fzerox/core/envs/rewards/reward_main/bounds.py
from __future__ import annotations

from rl_fzerox.core.envs.rewards.reward_main.progress import FrontierReward
from rl_fzerox.core.envs.rewards.shared_weights import SharedRewardWeights


def cap_outside_bounds_reentry_reward(
    frontier_reward: FrontierReward,
    *,
    weights: SharedRewardWeights,
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
        energy_gain_reward=0.0,
    )
