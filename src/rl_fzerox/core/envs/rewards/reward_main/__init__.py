# src/rl_fzerox/core/envs/rewards/reward_main/__init__.py
"""Canonical `reward_main` profile facade.

The public surface is intentionally just the tracker and weight dataclass.
Individual reward terms are split into nearby modules so the tracker remains
focused on episode orchestration.
"""

from rl_fzerox.core.envs.rewards.reward_main.tracker import RewardMainTracker
from rl_fzerox.core.envs.rewards.reward_main.weights import RewardMainWeights

__all__ = ["RewardMainTracker", "RewardMainWeights"]
