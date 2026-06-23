# src/rl_fzerox/ui/watch/runtime/career_mode/loop/commands.py
from __future__ import annotations

import time
from dataclasses import dataclass

from fzerox_emulator import SpinRequest
from rl_fzerox.core.policy.auxiliary_state import AuxiliaryStateTargetName
from rl_fzerox.ui.watch.runtime.career_mode.session import CareerModeRuntimeSession
from rl_fzerox.ui.watch.runtime.career_mode.timing import (
    native_frame_seconds,
    set_session_control_timing,
)
from rl_fzerox.ui.watch.runtime.ipc import WorkerCommandBatch
from rl_fzerox.ui.watch.runtime.policy.cnn import CnnActivationNormalizationMode
from rl_fzerox.ui.watch.runtime.timing import RateMeter, _adjust_control_fps, _target_seconds


@dataclass(frozen=True, slots=True)
class ControlTimingUpdate:
    target_control_fps: float | None
    target_control_seconds: float | None
    native_frame_seconds: float | None
    next_step_time: float


@dataclass(frozen=True, slots=True)
class CommandRuntimeUpdate:
    """Pure UI/control state derived from one drained worker-command batch."""

    cnn_visualization_enabled: bool
    auxiliary_visualization_enabled: bool
    live_visualization_enabled: bool
    last_live_series_publish_time: float
    cnn_normalization: CnnActivationNormalizationMode
    deterministic_policy: bool
    manual_control_enabled: bool
    spin_request: SpinRequest


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


def apply_runtime_command_state(
    *,
    commands: WorkerCommandBatch,
    auxiliary_target_names: tuple[AuxiliaryStateTargetName, ...],
    previous_live_visualization_enabled: bool,
    last_live_series_publish_time: float,
    deterministic_policy: bool,
    policy_owns_control: bool,
    active_policy_started: bool,
) -> CommandRuntimeUpdate:
    """Return loop-local control flags implied by one command batch."""

    live_visualization_enabled = commands.live_visualization_enabled
    if live_visualization_enabled != previous_live_visualization_enabled:
        last_live_series_publish_time = 0.0

    next_manual_control_enabled = (
        commands.manual_control_enabled if policy_owns_control and active_policy_started else False
    )
    spin_request: SpinRequest = commands.spin_request if next_manual_control_enabled else "none"

    return CommandRuntimeUpdate(
        cnn_visualization_enabled=commands.cnn_visualization_enabled,
        auxiliary_visualization_enabled=(
            bool(auxiliary_target_names) and commands.auxiliary_visualization_enabled
        ),
        live_visualization_enabled=live_visualization_enabled,
        last_live_series_publish_time=last_live_series_publish_time,
        cnn_normalization=commands.cnn_normalization,
        deterministic_policy=(
            not deterministic_policy
            if commands.toggle_deterministic_policy
            else deterministic_policy
        ),
        manual_control_enabled=next_manual_control_enabled,
        spin_request=spin_request,
    )
