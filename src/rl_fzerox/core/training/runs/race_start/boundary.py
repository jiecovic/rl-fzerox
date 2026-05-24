# src/rl_fzerox/core/training/runs/race_start/boundary.py
from __future__ import annotations

from rl_fzerox.core.domain.race_difficulty import race_difficulty_raw_value
from rl_fzerox.core.training.runs.race_start.models import RaceStartVariant


def race_start_gp_difficulty_raw_value(variant: RaceStartVariant) -> int:
    if variant.mode != "gp_race" or variant.gp_difficulty is None:
        return -1
    return race_difficulty_raw_value(variant.gp_difficulty)
