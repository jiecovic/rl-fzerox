# src/rl_fzerox/core/envs/rewards/race_v4/__init__.py
"""Milestone/time-pressure reward profile for reward-shaping experiments."""

from rl_fzerox.core.envs.rewards.race_v4.tracker import RaceV4RewardTracker
from rl_fzerox.core.envs.rewards.race_v4.weights import RaceV4RewardWeights

__all__ = ["RaceV4RewardTracker", "RaceV4RewardWeights"]
