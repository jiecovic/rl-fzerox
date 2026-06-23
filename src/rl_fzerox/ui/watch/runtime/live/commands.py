# src/rl_fzerox/ui/watch/runtime/live/commands.py
from __future__ import annotations

import time
from dataclasses import dataclass

from fzerox_emulator import SpinRequest
from rl_fzerox.core.policy.auxiliary_state import AuxiliaryStateTargetName
from rl_fzerox.ui.watch.runtime.ipc import WorkerCommandBatch
from rl_fzerox.ui.watch.runtime.policy.cnn import CnnActivationNormalizationMode
from rl_fzerox.ui.watch.runtime.timing import RateMeter, _adjust_control_fps, _target_seconds


@dataclass(frozen=True, slots=True)
class LiveCommandStateUpdate:
    cnn_visualization_enabled: bool
    auxiliary_visualization_enabled: bool
    live_visualization_enabled: bool
    live_visualization_changed: bool
    last_live_series_publish_time: float
    cnn_normalization: CnnActivationNormalizationMode
    deterministic_policy: bool
    deterministic_policy_changed: bool
    manual_control_enabled: bool
    spin_request: SpinRequest


@dataclass(frozen=True, slots=True)
class LiveControlTimingUpdate:
    target_control_fps: float | None
    target_control_seconds: float | None
    next_step_time: float


def apply_live_command_state(
    *,
    commands: WorkerCommandBatch,
    auxiliary_target_names: tuple[AuxiliaryStateTargetName, ...],
    previous_live_visualization_enabled: bool,
    last_live_series_publish_time: float,
    deterministic_policy: bool,
    policy_available: bool,
) -> LiveCommandStateUpdate:
    """Return live-worker UI/control state implied by one command batch."""

    live_visualization_enabled = commands.live_visualization_enabled
    live_visualization_changed = live_visualization_enabled != previous_live_visualization_enabled
    if live_visualization_changed:
        last_live_series_publish_time = 0.0

    manual_control_enabled = commands.manual_control_enabled if policy_available else True
    spin_request: SpinRequest = commands.spin_request if manual_control_enabled else "none"
    deterministic_policy_changed = commands.toggle_deterministic_policy and policy_available

    return LiveCommandStateUpdate(
        cnn_visualization_enabled=commands.cnn_visualization_enabled,
        auxiliary_visualization_enabled=(
            bool(auxiliary_target_names) and commands.auxiliary_visualization_enabled
        ),
        live_visualization_enabled=live_visualization_enabled,
        live_visualization_changed=live_visualization_changed,
        last_live_series_publish_time=last_live_series_publish_time,
        cnn_normalization=commands.cnn_normalization,
        deterministic_policy=(
            not deterministic_policy if deterministic_policy_changed else deterministic_policy
        ),
        deterministic_policy_changed=deterministic_policy_changed,
        manual_control_enabled=manual_control_enabled,
        spin_request=spin_request,
    )


def apply_live_control_timing_command(
    *,
    commands: WorkerCommandBatch,
    control_rate: RateMeter,
    native_control_fps: float,
    target_control_fps: float | None,
) -> LiveControlTimingUpdate | None:
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

    control_rate.trim_to_recent()
    return LiveControlTimingUpdate(
        target_control_fps=next_target_control_fps,
        target_control_seconds=_target_seconds(next_target_control_fps),
        next_step_time=time.perf_counter(),
    )
