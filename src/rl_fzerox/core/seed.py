# src/rl_fzerox/core/seed.py
from __future__ import annotations

import random

import numpy as np


def seed_process(seed: int | None) -> None:
    # The current runtime is deterministic on the emulator side. The master seed
    # therefore owns Python-side randomness and future stochastic components.
    if seed is None:
        return

    random.seed(seed)
    np.random.seed(seed)


def episode_seed(master_seed: int | None, episode_index: int) -> int | None:
    if master_seed is None:
        return None
    return master_seed + episode_index
