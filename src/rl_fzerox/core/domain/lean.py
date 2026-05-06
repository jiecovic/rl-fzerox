from __future__ import annotations

from typing import Final, Literal, TypeAlias

LeanMode: TypeAlias = Literal[
    "minimum_hold",
    "release_cooldown",
    "timer_assist",
    "raw",
]

DEFAULT_LEAN_MODE: Final[LeanMode] = "release_cooldown"
