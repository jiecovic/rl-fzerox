# src/rl_fzerox/ui/watch/runtime/x_cup.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fzerox_emulator import (
    JOYPAD_L2,
    JOYPAD_R,
    JOYPAD_RIGHT,
    JOYPAD_START,
    ControllerState,
    Emulator,
    FZeroXTelemetry,
    joypad_mask,
)
from fzerox_emulator._native import JOYPAD_L
from rl_fzerox.core.config.schema import WatchAppConfig


@dataclass(frozen=True, slots=True)
class XCupBootstrapTiming:
    """Measured input timings for the watch-only X Cup boot script."""

    boot_frames: int = 300
    main_menu_ready_frames: int = 90
    post_unlock_settle_frames: int = 120
    start_hold_frames: int = 2
    start_settle_frames: int = 38
    menu_hold_frames: int = 8
    menu_settle_frames: int = 60
    mode_press_limit: int = 12
    race_mode_poll_frames: int = 120
    race_init_frame_limit: int = 1_500
    race_intro_fresh_init_floor: int = 100
    camera_sync_ready_timer: int = 160


@dataclass(frozen=True, slots=True)
class XCupBootstrapInfo:
    """Flat metadata attached to watch snapshots for X Cup sessions."""

    course_index: int = 48
    course_id: str = "x_cup"
    course_name: str = "X"
    display_name: str = "X Cup"
    mode: str = "gp_race"
    vehicle: str = "blue_falcon"
    vehicle_name: str = "Blue Falcon"
    engine_setting: str = "balanced"

    def info(self) -> dict[str, object]:
        return {
            "track_course_index": self.course_index,
            "track_course_id": self.course_id,
            "track_course_name": self.course_name,
            "track_display_name": self.display_name,
            "track_mode": self.mode,
            "track_vehicle": self.vehicle,
            "track_vehicle_name": self.vehicle_name,
            "track_engine_setting": self.engine_setting,
            "watch_x_cup_enabled": True,
        }


X_CUP_BOOTSTRAP_TIMING = XCupBootstrapTiming()
X_CUP_BOOTSTRAP_INFO = XCupBootstrapInfo()
_X_CUP_MENU_INDEX = 48

# The USA ROM path exposes X Cup only after the standard unlock-everything
# code is entered on the main mode-select screen.
_UNLOCK_EVERYTHING_SEQUENCE: tuple[ControllerState, ...] = (
    ControllerState(joypad_mask=joypad_mask(JOYPAD_L)),
    ControllerState(joypad_mask=joypad_mask(JOYPAD_L2)),
    ControllerState(joypad_mask=joypad_mask(JOYPAD_R)),
    ControllerState(right_stick_y=-1.0),
    ControllerState(right_stick_y=1.0),
    ControllerState(right_stick_x=-1.0),
    ControllerState(right_stick_x=1.0),
    ControllerState(joypad_mask=joypad_mask(JOYPAD_START)),
)


def materialize_x_cup_watch_baseline(config: WatchAppConfig) -> dict[str, object]:
    """Create one watch-session baseline by cold-booting into GP X Cup.

    This is deliberately watch-only. The current baseline materializer and
    training reset stack are still built around built-in Time Attack tracks.
    """

    baseline_state_path = config.emulator.baseline_state_path
    if baseline_state_path is None:
        raise RuntimeError("watch.x_cup.enabled requires a writable watch baseline path")

    bootstrap_runtime_dir = _bootstrap_runtime_dir(config.emulator.runtime_dir)
    if bootstrap_runtime_dir is not None:
        bootstrap_runtime_dir.mkdir(parents=True, exist_ok=True)

    emulator = Emulator(
        core_path=config.emulator.core_path,
        rom_path=config.emulator.rom_path,
        runtime_dir=bootstrap_runtime_dir,
        baseline_state_path=None,
        renderer=config.emulator.renderer,
    )
    try:
        _boot_into_x_cup_gp_race(emulator)
        baseline_state_path.parent.mkdir(parents=True, exist_ok=True)
        emulator.capture_current_as_baseline(baseline_state_path)
    finally:
        emulator.close()
    return X_CUP_BOOTSTRAP_INFO.info()


def _bootstrap_runtime_dir(runtime_dir: Path | None) -> Path | None:
    if runtime_dir is None:
        return None
    return runtime_dir / "x_cup_bootstrap"


def _boot_into_x_cup_gp_race(emulator: Emulator) -> None:
    emulator.reset()
    emulator.step_frames(X_CUP_BOOTSTRAP_TIMING.boot_frames, capture_video=False)
    for _ in range(4):
        _tap_start(emulator)
    emulator.step_frames(X_CUP_BOOTSTRAP_TIMING.main_menu_ready_frames, capture_video=False)
    for control_state in _UNLOCK_EVERYTHING_SEQUENCE:
        _tap_state(emulator, control_state)
    emulator.step_frames(X_CUP_BOOTSTRAP_TIMING.post_unlock_settle_frames, capture_video=False)

    _press_start_until_mode(emulator, target_mode="course_select")
    for _ in range(4):
        _tap_menu_right(emulator)
    _validate_x_cup_selection(emulator)
    _press_start_until_mode(emulator, target_mode="machine_select")
    _press_start_until_mode(emulator, target_mode="machine_settings")
    _press_start_until_x_cup_race_mode(emulator)
    _wait_for_x_cup_race_intro_window(emulator)


def _validate_x_cup_selection(emulator: Emulator) -> None:
    telemetry = emulator.try_read_telemetry()
    if telemetry is None:
        raise RuntimeError("X Cup bootstrap could not confirm the course-select telemetry state")
    if int(telemetry.course_index) != _X_CUP_MENU_INDEX:
        raise RuntimeError(
            "X Cup bootstrap did not land on the expected course slot; "
            f"expected course_index={_X_CUP_MENU_INDEX}, got {telemetry.course_index}"
        )


def _press_start_until_mode(
    emulator: Emulator,
    *,
    target_mode: str,
    require_race_mode: bool = False,
) -> None:
    for _ in range(X_CUP_BOOTSTRAP_TIMING.mode_press_limit):
        telemetry = emulator.try_read_telemetry()
        if telemetry is not None and telemetry.game_mode_name == target_mode:
            if not require_race_mode or telemetry.in_race_mode:
                return
        _tap_start(emulator)

    telemetry = emulator.try_read_telemetry()
    current_mode = None if telemetry is None else telemetry.game_mode_name
    raise RuntimeError(
        f"X Cup bootstrap did not reach {target_mode!r}; last mode was {current_mode!r}"
    )


def _press_start_until_x_cup_race_mode(emulator: Emulator) -> None:
    """Enter GP X Cup race mode without burning through the intro countdown."""

    for _ in range(X_CUP_BOOTSTRAP_TIMING.mode_press_limit):
        telemetry = emulator.try_read_telemetry()
        if _is_x_cup_race_telemetry(telemetry):
            return

        emulator.set_controller_state(ControllerState(joypad_mask=joypad_mask(JOYPAD_START)))
        emulator.step_frames(X_CUP_BOOTSTRAP_TIMING.start_hold_frames, capture_video=False)
        emulator.set_controller_state(ControllerState())
        for _ in range(X_CUP_BOOTSTRAP_TIMING.race_mode_poll_frames):
            telemetry = emulator.try_read_telemetry()
            if _is_x_cup_race_telemetry(telemetry):
                return
            emulator.step_frames(1, capture_video=False)

    telemetry = emulator.try_read_telemetry()
    current_mode = None if telemetry is None else telemetry.game_mode_name
    raise RuntimeError(
        "X Cup bootstrap did not enter 'gp_race'; "
        f"last mode was {current_mode!r}"
    )


def _wait_for_x_cup_race_intro_window(emulator: Emulator) -> None:
    """Capture late enough that the normal camera sync taps are accepted."""

    last_summary = "telemetry unavailable"
    saw_new_race_init = False
    for _ in range(X_CUP_BOOTSTRAP_TIMING.race_init_frame_limit):
        telemetry = emulator.try_read_telemetry()
        if telemetry is None:
            emulator.step_frames(1, capture_video=False)
            continue

        last_summary = (
            f"mode={telemetry.game_mode_name!r}, course={telemetry.course_index}, "
            f"intro={telemetry.race_intro_timer}"
        )
        if not _is_x_cup_race_telemetry(telemetry):
            emulator.step_frames(1, capture_video=False)
            continue
        if int(telemetry.race_intro_timer) > X_CUP_BOOTSTRAP_TIMING.race_intro_fresh_init_floor:
            saw_new_race_init = True
        if saw_new_race_init and (
            int(telemetry.race_intro_timer) <= X_CUP_BOOTSTRAP_TIMING.camera_sync_ready_timer
        ):
            return
        emulator.step_frames(1, capture_video=False)

    raise RuntimeError(
        "X Cup bootstrap did not reach a fresh GP intro window within "
        f"{X_CUP_BOOTSTRAP_TIMING.race_init_frame_limit} frames ({last_summary})"
    )


def _is_x_cup_race_telemetry(telemetry: FZeroXTelemetry | None) -> bool:
    if telemetry is None:
        return False
    return bool(
        telemetry.in_race_mode
        and telemetry.game_mode_name == "gp_race"
        and int(telemetry.course_index) == _X_CUP_MENU_INDEX
    )


def _tap_start(emulator: Emulator) -> None:
    _tap_state(
        emulator,
        ControllerState(joypad_mask=joypad_mask(JOYPAD_START)),
        hold_frames=X_CUP_BOOTSTRAP_TIMING.start_hold_frames,
        settle_frames=X_CUP_BOOTSTRAP_TIMING.start_settle_frames,
    )


def _tap_menu_right(emulator: Emulator) -> None:
    _tap_state(
        emulator,
        ControllerState(joypad_mask=joypad_mask(JOYPAD_RIGHT)),
        hold_frames=X_CUP_BOOTSTRAP_TIMING.menu_hold_frames,
        settle_frames=X_CUP_BOOTSTRAP_TIMING.menu_settle_frames,
    )


def _tap_state(
    emulator: Emulator,
    control_state: ControllerState,
    *,
    hold_frames: int = X_CUP_BOOTSTRAP_TIMING.menu_hold_frames,
    settle_frames: int = X_CUP_BOOTSTRAP_TIMING.menu_settle_frames,
) -> None:
    emulator.set_controller_state(control_state)
    emulator.step_frames(hold_frames, capture_video=False)
    emulator.set_controller_state(ControllerState())
    emulator.step_frames(settle_frames, capture_video=False)
