# src/rl_fzerox/core/domain/shoulder_slide.py
from __future__ import annotations

from typing import Final, Literal, TypeAlias

ShoulderSlideMode: TypeAlias = Literal[
    "minimum_hold",
    "release_cooldown",
    "timer_assist",
]

SHOULDER_SLIDE_MODE_MINIMUM_HOLD: Final[ShoulderSlideMode] = "minimum_hold"
SHOULDER_SLIDE_MODE_RELEASE_COOLDOWN: Final[ShoulderSlideMode] = "release_cooldown"
SHOULDER_SLIDE_MODE_TIMER_ASSIST: Final[ShoulderSlideMode] = "timer_assist"
DEFAULT_SHOULDER_SLIDE_MODE: Final[ShoulderSlideMode] = SHOULDER_SLIDE_MODE_RELEASE_COOLDOWN
