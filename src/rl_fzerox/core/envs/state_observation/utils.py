# src/rl_fzerox/core/envs/state_observation/utils.py
from __future__ import annotations


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))
