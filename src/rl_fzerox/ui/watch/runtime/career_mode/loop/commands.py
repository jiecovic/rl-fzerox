# src/rl_fzerox/ui/watch/runtime/career_mode/loop/commands.py
from __future__ import annotations

import time
from dataclasses import dataclass

from rl_fzerox.ui.watch.runtime.career_mode.session import CareerModeRuntimeSession
from rl_fzerox.ui.watch.runtime.career_mode.timing import (
    native_frame_seconds,
    set_session_control_timing,
)
from rl_fzerox.ui.watch.runtime.ipc import WorkerCommandBatch
from rl_fzerox.ui.watch.runtime.timing import RateMeter, _adjust_control_fps, _target_seconds


@dataclass(frozen=True, slots=True)
class ControlTimingUpdate:
    target_control_fps: float | None
    target_control_seconds: float | None
    native_frame_seconds: float | None
    next_step_time: float


def apply_control_timing_command(
    *,
    commands: WorkerCommandBatch,
    session: CareerModeRuntimeSession,
    control_rate: RateMeter,
    native_control_fps: float,
    target_control_fps: float | None,
) -> ControlTimingUpdate | None:
    if commands.reset_control_fps:
        next_target_control_fps = native_control_fps
    elif commands.control_fps_delta:
        next_target_control_fps = _adjust_control_fps(
            target_control_fps,
            commands.control_fps_delta,
            native_control_fps=native_control_fps,
        )
    else:
        return None

    next_target_control_seconds = _target_seconds(next_target_control_fps)
    next_native_frame_seconds = native_frame_seconds(next_target_control_seconds)
    set_session_control_timing(
        session,
        target_control_fps=next_target_control_fps,
        target_control_seconds=next_target_control_seconds,
    )
    control_rate.reset()
    return ControlTimingUpdate(
        target_control_fps=next_target_control_fps,
        target_control_seconds=next_target_control_seconds,
        native_frame_seconds=next_native_frame_seconds,
        next_step_time=time.perf_counter(),
    )
