# src/rl_fzerox/core/envs/engine/controls/episode_dropout.py
from __future__ import annotations

import random


def sample_episode_lean_mask(*, probability: float, seed: int | None) -> bool:
    """Sample whether lean should be masked for one whole episode."""

    if probability <= 0.0:
        return False
    if probability >= 1.0:
        return True
    if seed is None:
        return random.random() < probability
    return random.Random(seed).random() < probability
