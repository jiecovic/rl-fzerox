# src/fzerox_emulator/control/spin.py
"""High-level control macro request values understood by the native host."""

from __future__ import annotations

from typing import Literal

type SpinRequest = Literal["none", "left", "right"]

DEFAULT_SPIN_COOLDOWN_FRAMES = 120

SPIN_REQUESTS: tuple[SpinRequest, ...] = ("none", "left", "right")


def spin_request_from_index(index: int) -> SpinRequest:
    """Map one 3-way policy branch value to a native spin request."""

    if not 0 <= index < len(SPIN_REQUESTS):
        raise ValueError(f"Invalid spin index {index}")
    return SPIN_REQUESTS[index]
