# tests/ui/test_live_worker_commands.py
from __future__ import annotations

import pytest

from fzerox_emulator import RaceControlState, SpinRequest
from rl_fzerox.ui.watch.runtime.ipc import WorkerCommandBatch
from rl_fzerox.ui.watch.runtime.live.commands import (
    apply_live_command_state,
    apply_live_control_timing_command,
)
from rl_fzerox.ui.watch.runtime.policy.cnn import DEFAULT_CNN_ACTIVATION_NORMALIZATION
from rl_fzerox.ui.watch.runtime.timing import RateMeter


def test_live_command_state_forces_manual_control_without_policy() -> None:
    update = apply_live_command_state(
        commands=_command_batch(
            manual_control_enabled=False,
            spin_request="left",
        ),
        auxiliary_target_names=(),
        previous_live_visualization_enabled=False,
        last_live_series_publish_time=12.0,
        deterministic_policy=True,
        policy_available=False,
    )

    assert update.manual_control_enabled is True
    assert update.spin_request == "left"
    assert update.deterministic_policy is True
    assert update.deterministic_policy_changed is False


def test_live_command_state_gates_manual_control_with_policy() -> None:
    update = apply_live_command_state(
        commands=_command_batch(
            manual_control_enabled=False,
            spin_request="left",
        ),
        auxiliary_target_names=("vehicle_state.speed_norm",),
        previous_live_visualization_enabled=False,
        last_live_series_publish_time=12.0,
        deterministic_policy=True,
        policy_available=True,
    )

    assert update.auxiliary_visualization_enabled is False
    assert update.manual_control_enabled is False
    assert update.spin_request == "none"


def test_live_command_state_resets_live_series_and_toggles_policy() -> None:
    update = apply_live_command_state(
        commands=_command_batch(
            auxiliary_visualization_enabled=True,
            live_visualization_enabled=True,
            toggle_deterministic_policy=True,
        ),
        auxiliary_target_names=("vehicle_state.speed_norm",),
        previous_live_visualization_enabled=False,
        last_live_series_publish_time=12.0,
        deterministic_policy=True,
        policy_available=True,
    )

    assert update.auxiliary_visualization_enabled is True
    assert update.live_visualization_changed is True
    assert update.last_live_series_publish_time == 0.0
    assert update.deterministic_policy is False
    assert update.deterministic_policy_changed is True


def test_live_control_timing_reset_uses_native_fps() -> None:
    update = apply_live_control_timing_command(
        commands=_command_batch(reset_control_fps=True),
        control_rate=RateMeter(window=4),
        native_control_fps=60.0,
        target_control_fps=30.0,
    )

    assert update is not None
    assert update.target_control_fps == 60.0
    assert update.target_control_seconds == pytest.approx(1.0 / 60.0)
    assert update.next_step_time > 0.0


def test_live_control_timing_delta_adjusts_current_target() -> None:
    update = apply_live_control_timing_command(
        commands=_command_batch(control_fps_delta=-1),
        control_rate=RateMeter(window=4),
        native_control_fps=60.0,
        target_control_fps=60.0,
    )

    assert update is not None
    assert update.target_control_fps == 55.0
    assert update.target_control_seconds == pytest.approx(1.0 / 55.0)


def _command_batch(
    *,
    auxiliary_visualization_enabled: bool = False,
    live_visualization_enabled: bool = False,
    manual_control_enabled: bool = False,
    spin_request: SpinRequest = "none",
    toggle_deterministic_policy: bool = False,
    control_fps_delta: int = 0,
    reset_control_fps: bool = False,
) -> WorkerCommandBatch:
    return WorkerCommandBatch(
        quit_requested=False,
        paused=False,
        step_requests=0,
        save_requests=0,
        reset_mode=None,
        jump_course_id=None,
        difficulty_delta=0,
        toggle_deterministic_policy=toggle_deterministic_policy,
        manual_control_enabled=manual_control_enabled,
        toggle_current_course_lock=False,
        toggle_zeroed_state_feature_name=None,
        control_fps_delta=control_fps_delta,
        reset_control_fps=reset_control_fps,
        cnn_visualization_enabled=False,
        auxiliary_visualization_enabled=auxiliary_visualization_enabled,
        live_visualization_enabled=live_visualization_enabled,
        cnn_normalization=DEFAULT_CNN_ACTIVATION_NORMALIZATION,
        spin_request=spin_request,
        control_state=RaceControlState(),
    )
