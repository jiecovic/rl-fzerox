# tests/ui/test_watch_runtime_commands.py
"""Watch runtime IPC tests for coalescing viewer commands.

These cases verify how queued UI commands collapse into one worker command batch
before the live runtime loop applies them.
"""

from fzerox_emulator import RaceControlState
from rl_fzerox.ui.watch.runtime.courses.commands import apply_course_navigation_commands
from rl_fzerox.ui.watch.runtime.ipc import ViewerCommand, drain_worker_commands
from tests.ui.watch_runtime_ipc_support import (
    _CommandQueue,
    _sample_rotation,
    _SequentialResetEnv,
)


def test_drain_worker_commands_coalesces_reset_mode() -> None:
    command_queue = _CommandQueue([ViewerCommand(reset_mode="current")])

    commands, paused, control_state = drain_worker_commands(
        command_queue,
        paused=False,
        control_state=RaceControlState(),
    )

    assert commands.reset_mode == "current"
    assert paused is False
    assert control_state == RaceControlState()


def test_drain_worker_commands_last_reset_mode_wins() -> None:
    command_queue = _CommandQueue(
        [
            ViewerCommand(reset_mode="current"),
            ViewerCommand(reset_mode="next"),
        ]
    )

    commands, _, _ = drain_worker_commands(
        command_queue,
        paused=False,
        control_state=RaceControlState(),
    )

    assert commands.reset_mode == "next"


def test_drain_worker_commands_coalesces_deterministic_toggle_parity() -> None:
    command_queue = _CommandQueue(
        [
            ViewerCommand(toggle_deterministic_policy=True),
            ViewerCommand(toggle_deterministic_policy=True),
            ViewerCommand(toggle_deterministic_policy=True),
        ]
    )

    commands, paused, control_state = drain_worker_commands(
        command_queue,
        paused=False,
        control_state=RaceControlState(),
    )

    assert commands.toggle_deterministic_policy is True
    assert paused is False
    assert control_state == RaceControlState()


def test_drain_worker_commands_preserves_cnn_visualization_state_without_commands() -> None:
    command_queue = _CommandQueue([])

    commands, paused, control_state = drain_worker_commands(
        command_queue,
        paused=False,
        control_state=RaceControlState(),
        cnn_visualization_enabled=True,
    )

    assert commands.cnn_visualization_enabled is True
    assert paused is False
    assert control_state == RaceControlState()


def test_drain_worker_commands_updates_cnn_visualization_state() -> None:
    command_queue = _CommandQueue([ViewerCommand(cnn_visualization_enabled=False)])

    commands, _, _ = drain_worker_commands(
        command_queue,
        paused=False,
        control_state=RaceControlState(),
        cnn_visualization_enabled=True,
    )

    assert commands.cnn_visualization_enabled is False


def test_drain_worker_commands_updates_cnn_normalization() -> None:
    command_queue = _CommandQueue([ViewerCommand(cnn_normalization="layer_percentile")])

    commands, _, _ = drain_worker_commands(
        command_queue,
        paused=False,
        control_state=RaceControlState(),
        cnn_normalization="channel",
    )

    assert commands.cnn_normalization == "layer_percentile"


def test_drain_worker_commands_toggles_manual_control_state() -> None:
    command_queue = _CommandQueue([ViewerCommand(toggle_manual_control=True)])

    commands, _, _ = drain_worker_commands(
        command_queue,
        paused=False,
        control_state=RaceControlState(),
        manual_control_enabled=False,
    )

    assert commands.manual_control_enabled is True


def test_drain_worker_commands_coalesces_course_jump_selection() -> None:
    command_queue = _CommandQueue(
        [
            ViewerCommand(jump_course_id="mute_city"),
            ViewerCommand(jump_course_id="silence"),
        ]
    )

    commands, _, _ = drain_worker_commands(
        command_queue,
        paused=False,
        control_state=RaceControlState(),
    )

    assert commands.jump_course_id == "silence"


def test_drain_worker_commands_accumulates_difficulty_delta() -> None:
    command_queue = _CommandQueue(
        [
            ViewerCommand(difficulty_delta=1),
            ViewerCommand(difficulty_delta=-1),
            ViewerCommand(difficulty_delta=1),
        ]
    )

    commands, _, _ = drain_worker_commands(
        command_queue,
        paused=False,
        control_state=RaceControlState(),
    )

    assert commands.difficulty_delta == 1


def test_difficulty_command_forces_reset_to_same_course_on_next_difficulty() -> None:
    env = _SequentialResetEnv()
    rotation = _sample_rotation()
    command_queue = _CommandQueue([ViewerCommand(difficulty_delta=1)])
    commands, _, _ = drain_worker_commands(
        command_queue,
        paused=False,
        control_state=RaceControlState(),
    )
    info: dict[str, object] = {
        "track_reset_target_key": "mute_city#difficulty=novice",
        "track_reset_course_key": "mute_city",
        "track_gp_difficulty": "novice",
    }
    reset_info = dict(info)

    result = apply_course_navigation_commands(
        commands,
        env=env,
        info=info,
        reset_info=reset_info,
        rotation=rotation,
        selected_reset_target_key="mute_city#difficulty=novice",
        locked_reset_target_key=None,
    )

    assert result.reset_requested is True
    assert result.selected_reset_target_key == "mute_city#difficulty=expert"
    assert env.next_courses == ["mute_city#difficulty=expert"]
    assert reset_info["watch_selected_gp_difficulty"] == "expert"


def test_drain_worker_commands_coalesces_current_course_lock_toggle() -> None:
    command_queue = _CommandQueue(
        [
            ViewerCommand(toggle_current_course_lock=True),
            ViewerCommand(toggle_current_course_lock=True),
        ]
    )

    commands, _, _ = drain_worker_commands(
        command_queue,
        paused=False,
        control_state=RaceControlState(),
    )

    assert commands.toggle_current_course_lock is True


def test_drain_worker_commands_coalesces_state_feature_toggle() -> None:
    command_queue = _CommandQueue(
        [
            ViewerCommand(toggle_zeroed_state_feature_name="vehicle_state.speed"),
            ViewerCommand(toggle_zeroed_state_feature_name="course_context"),
        ]
    )

    commands, _, _ = drain_worker_commands(
        command_queue,
        paused=False,
        control_state=RaceControlState(),
    )

    assert commands.toggle_zeroed_state_feature_name == "course_context"


def test_drain_worker_commands_coalesces_control_fps_reset() -> None:
    command_queue = _CommandQueue(
        [
            ViewerCommand(control_fps_delta=1),
            ViewerCommand(reset_control_fps=True),
        ]
    )

    commands, _, _ = drain_worker_commands(
        command_queue,
        paused=False,
        control_state=RaceControlState(),
    )

    assert commands.control_fps_delta == 1
    assert commands.reset_control_fps is True


def test_drain_worker_commands_coalesces_spin_request() -> None:
    command_queue = _CommandQueue([ViewerCommand(spin_request="left")])

    commands, _, _ = drain_worker_commands(
        command_queue,
        paused=False,
        control_state=RaceControlState(),
    )

    assert commands.spin_request == "left"


def test_drain_worker_commands_last_non_idle_spin_request_wins() -> None:
    command_queue = _CommandQueue(
        [
            ViewerCommand(spin_request="left"),
            ViewerCommand(spin_request="none"),
            ViewerCommand(spin_request="right"),
        ]
    )

    commands, _, _ = drain_worker_commands(
        command_queue,
        paused=False,
        control_state=RaceControlState(),
    )

    assert commands.spin_request == "right"


def test_drain_worker_commands_preserves_held_spin_request_without_commands() -> None:
    command_queue = _CommandQueue([])

    commands, _, _ = drain_worker_commands(
        command_queue,
        paused=False,
        control_state=RaceControlState(),
        spin_request="left",
    )

    assert commands.spin_request == "left"


def test_drain_worker_commands_clears_released_spin_request() -> None:
    command_queue = _CommandQueue([ViewerCommand(spin_request="none")])

    commands, _, _ = drain_worker_commands(
        command_queue,
        paused=False,
        control_state=RaceControlState(),
        spin_request="left",
    )

    assert commands.spin_request == "none"
