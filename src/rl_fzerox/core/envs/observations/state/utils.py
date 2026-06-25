# src/rl_fzerox/core/envs/observations/state/utils.py
"""Small numeric helpers for state-vector feature extraction.

Keep only observation-state-local helpers here. Shared env math should live in a
more explicit module with a named responsibility.
"""
from __future__ import annotations


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))
