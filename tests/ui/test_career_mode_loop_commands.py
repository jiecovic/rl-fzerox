# tests/ui/test_career_mode_loop_commands.py
from __future__ import annotations

from fzerox_emulator import RaceControlState, SpinRequest
from rl_fzerox.ui.watch.runtime.career_mode.loop.commands import (
    apply_runtime_command_state,
)
from rl_fzerox.ui.watch.runtime.ipc import WorkerCommandBatch
from rl_fzerox.ui.watch.runtime.policy.cnn import DEFAULT_CNN_ACTIVATION_NORMALIZATION


def test_runtime_command_state_resets_live_series_publish_time_on_toggle() -> None:
    update = apply_runtime_command_state(
        commands=_command_batch(live_visualization_enabled=True),
        auxiliary_target_names=(),
        previous_live_visualization_enabled=False,
        last_live_series_publish_time=123.0,
        deterministic_policy=True,
        policy_owns_control=False,
        active_policy_started=False,
    )

    assert update.live_visualization_enabled is True
    assert update.last_live_series_publish_time == 0.0


def test_runtime_command_state_gates_auxiliary_and_manual_control() -> None:
    no_targets = apply_runtime_command_state(
        commands=_command_batch(
            auxiliary_visualization_enabled=True,
            manual_control_enabled=True,
            spin_request="left",
        ),
        auxiliary_target_names=(),
        previous_live_visualization_enabled=False,
        last_live_series_publish_time=12.0,
        deterministic_policy=True,
        policy_owns_control=True,
        active_policy_started=True,
    )
    no_policy_control = apply_runtime_command_state(
        commands=_command_batch(
            auxiliary_visualization_enabled=True,
            manual_control_enabled=True,
            spin_request="left",
        ),
        auxiliary_target_names=("vehicle_state.speed_norm",),
        previous_live_visualization_enabled=False,
        last_live_series_publish_time=12.0,
        deterministic_policy=True,
        policy_owns_control=False,
        active_policy_started=True,
    )

    assert no_targets.auxiliary_visualization_enabled is False
    assert no_targets.manual_control_enabled is True
    assert no_targets.spin_request == "left"
    assert no_policy_control.auxiliary_visualization_enabled is True
    assert no_policy_control.manual_control_enabled is False
    assert no_policy_control.spin_request == "none"


def test_runtime_command_state_toggles_deterministic_policy() -> None:
    update = apply_runtime_command_state(
        commands=_command_batch(toggle_deterministic_policy=True),
        auxiliary_target_names=(),
        previous_live_visualization_enabled=False,
        last_live_series_publish_time=12.0,
        deterministic_policy=True,
        policy_owns_control=False,
        active_policy_started=False,
    )

    assert update.deterministic_policy is False


def _command_batch(
    *,
    auxiliary_visualization_enabled: bool = False,
    live_visualization_enabled: bool = False,
    manual_control_enabled: bool = False,
    spin_request: SpinRequest = "none",
    toggle_deterministic_policy: bool = False,
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
        control_fps_delta=0,
        reset_control_fps=False,
        cnn_visualization_enabled=False,
        auxiliary_visualization_enabled=auxiliary_visualization_enabled,
        live_visualization_enabled=live_visualization_enabled,
        cnn_normalization=DEFAULT_CNN_ACTIVATION_NORMALIZATION,
        spin_request=spin_request,
        control_state=RaceControlState(),
    )
