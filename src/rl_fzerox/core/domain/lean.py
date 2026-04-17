# src/rl_fzerox/core/domain/lean.py
from __future__ import annotations

from typing import Final, Literal, TypeAlias

LeanMode: TypeAlias = Literal[
    "minimum_hold",
    "release_cooldown",
    "timer_assist",
]

LEAN_MODE_MINIMUM_HOLD: Final[LeanMode] = "minimum_hold"
LEAN_MODE_RELEASE_COOLDOWN: Final[LeanMode] = "release_cooldown"
LEAN_MODE_TIMER_ASSIST: Final[LeanMode] = "timer_assist"
DEFAULT_LEAN_MODE: Final[LeanMode] = LEAN_MODE_RELEASE_COOLDOWN
