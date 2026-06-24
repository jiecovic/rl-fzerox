# src/rl_fzerox/core/domain/race/lean.py
from __future__ import annotations

from typing import Final, Literal

type LeanOutputMode = Literal[
    "three_way",
    "four_way_categorical",
    "independent_buttons",
]

type LeanMode = Literal[
    "minimum_hold",
    "release_cooldown",
    "timer_assist",
    "raw",
]

DEFAULT_LEAN_MODE: Final[LeanMode] = "release_cooldown"
