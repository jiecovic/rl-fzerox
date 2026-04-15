# src/rl_fzerox/core/envs/rewards/race_v3/__init__.py
"""Checkpoint-first reward profile for new reward-shaping experiments."""

from rl_fzerox.core.envs.rewards.race_v3.tracker import RaceV3RewardTracker
from rl_fzerox.core.envs.rewards.race_v3.weights import RaceV3RewardWeights

__all__ = ["RaceV3RewardTracker", "RaceV3RewardWeights"]
