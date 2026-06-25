# src/rl_fzerox/core/career_mode/controller/lifecycle/events.py
"""Small event payloads drained from the Career Mode controller.

Runtime loops read these objects after each controller step to close recordings
or reset the emulator at explicit FSM boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass

from rl_fzerox.core.career_mode.controller.lifecycle.recording import (
    CareerRecordingSegmentClose,
)


@dataclass(frozen=True, slots=True)
class CareerControllerLifecycleEvents:
    """Side-effect signals emitted by the Career Mode controller.

    The controller owns live Career Mode lifecycle decisions, but the watch
    runtime owns side effects such as writing recordings and resetting the
    emulator. Draining these signals together prevents callers from consuming a
    recording close while accidentally leaving the paired reset/exit decision
    for another branch.
    """

    recording_close: CareerRecordingSegmentClose | None
    emulator_reset_requested: bool
    has_active_attempt: bool
