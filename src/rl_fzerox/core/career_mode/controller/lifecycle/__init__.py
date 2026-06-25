# src/rl_fzerox/core/career_mode/controller/lifecycle/__init__.py
"""Lifecycle helper facade for controller-owned attempt side effects.

The FSM emits explicit recording and emulator-reset signals from these helpers;
runtime workers consume those signals instead of inferring lifecycle boundaries
from database state or frontend sync details.
"""

from __future__ import annotations

from rl_fzerox.core.career_mode.controller.lifecycle.events import (
    CareerControllerLifecycleEvents,
)
from rl_fzerox.core.career_mode.controller.lifecycle.post_race import PostRaceContinuation
from rl_fzerox.core.career_mode.controller.lifecycle.progress_sync import (
    CareerPostTerminalProgressSync,
)
from rl_fzerox.core.career_mode.controller.lifecycle.recording import (
    CareerRecordingSegmentClose,
    CareerRecordingSegmentStatus,
    CareerRecordingSegmentTracker,
    recording_status_from_attempt_status,
)
from rl_fzerox.core.career_mode.controller.lifecycle.terminal import (
    post_terminal_progress_screen,
    race_terminal_reason,
    terminal_info,
)

__all__ = [
    "CareerControllerLifecycleEvents",
    "CareerPostTerminalProgressSync",
    "CareerRecordingSegmentClose",
    "CareerRecordingSegmentStatus",
    "CareerRecordingSegmentTracker",
    "PostRaceContinuation",
    "post_terminal_progress_screen",
    "race_terminal_reason",
    "recording_status_from_attempt_status",
    "terminal_info",
]
