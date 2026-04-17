# src/rl_fzerox/core/boot.py
"""Deterministic emulator boot and race-continuation helpers.

This module owns the controller script that moves a freshly loaded F-Zero X
ROM from the title screen into the default first GP race, plus the shorter
path that advances an existing session from a terminal state into the next
race. It combines measured Start-button sequences with telemetry checks so
the env reset path can wait for real game-mode transitions instead of relying
only on fixed sleeps.

It intentionally stays separate from action adapters and training config:
the functions here mutate emulator state directly and are part of runtime
reset orchestration, not shared domain metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from fzerox_emulator import (
    JOYPAD_START,
    ControllerState,
    EmulatorBackend,
    FZeroXTelemetry,
    joypad_mask,
)
from fzerox_emulator.arrays import RgbFrame


@dataclass(frozen=True)
class BootConfig:
    """Deterministic timing and control settings for the boot script."""

    title_mode_name: str = "title"
    race_mode_name: str = "gp_race"
    start_control: ControllerState = ControllerState(joypad_mask=joypad_mask(JOYPAD_START))
    neutral_control: ControllerState = ControllerState()
    title_wait_frames: int = 240
    start_hold_frames: int = 2
    settle_frames: int = 60
    race_mode_wait_frames: int = 360
    pre_race_settle_frames: int = 180
    race_intro_wait_frames: int = 600
    next_race_max_start_presses: int = 12
    next_race_settle_frames: int = 45


BOOT_CONFIG = BootConfig()


class BootState(StrEnum):
    """Named phases for the deterministic first-race bootstrap path."""

    WAIT_FOR_TITLE = "wait_for_title"
    OPEN_MODE_MENU = "open_mode_menu"
    CONFIRM_DIFFICULTY = "confirm_difficulty"
    CONFIRM_COURSE = "confirm_course"
    ENTER_MACHINE_SELECT = "enter_machine_select"
    CONFIRM_MACHINE = "confirm_machine"
    CONFIRM_MACHINE_PROFILE = "confirm_machine_profile"
    WAIT_FOR_GP_RACE = "wait_for_gp_race"
    GP_RACE = "gp_race"


@dataclass(frozen=True)
class BootPhase:
    """One phase in the fixed Start-button bootstrap script."""

    state: BootState
    start_presses: int
    settle_frames: int


# This sequence was measured against the current USA ROM boot path with the
# default GP Race / Novice / Jack Cup / Blue Falcon selections.
BOOT_SEQUENCE: tuple[BootPhase, ...] = (
    BootPhase(BootState.OPEN_MODE_MENU, start_presses=4, settle_frames=BOOT_CONFIG.settle_frames),
    BootPhase(
        BootState.CONFIRM_DIFFICULTY,
        start_presses=2,
        settle_frames=BOOT_CONFIG.settle_frames,
    ),
    BootPhase(
        BootState.CONFIRM_COURSE,
        start_presses=2,
        settle_frames=BOOT_CONFIG.settle_frames,
    ),
    BootPhase(
        BootState.ENTER_MACHINE_SELECT,
        start_presses=2,
        settle_frames=BOOT_CONFIG.settle_frames,
    ),
    BootPhase(
        BootState.CONFIRM_MACHINE,
        start_presses=2,
        settle_frames=BOOT_CONFIG.settle_frames,
    ),
    BootPhase(
        BootState.CONFIRM_MACHINE_PROFILE,
        start_presses=4,
        settle_frames=BOOT_CONFIG.settle_frames,
    ),
)


def boot_into_first_race(backend: EmulatorBackend) -> tuple[RgbFrame, dict[str, object]]:
    """Drive the default USA boot path into the first Mute City grid.

    The input script still reflects the measured default menu path, but the
    settle points now prefer telemetry-validated game-mode transitions over
    blind fixed waits where the game exposes that state.
    """

    last_state = BootState.GP_RACE
    try:
        if not _wait_for_mode(
            backend,
            BOOT_CONFIG.title_mode_name,
            BOOT_CONFIG.title_wait_frames,
        ):
            raise RuntimeError(
                f"Timed out waiting for {BOOT_CONFIG.title_mode_name!r} mode during boot"
            )
        last_state = BootState.WAIT_FOR_TITLE
        for phase in BOOT_SEQUENCE:
            for _ in range(phase.start_presses):
                _press_start(backend)
                _settle_after_input(backend, phase.settle_frames)
            last_state = phase.state
        last_state = BootState.WAIT_FOR_GP_RACE
        if not _wait_for_gp_race(backend):
            raise RuntimeError(
                f"Timed out waiting for {BOOT_CONFIG.race_mode_name!r} mode during boot"
            )
        backend.step_frames(BOOT_CONFIG.pre_race_settle_frames)
    finally:
        backend.set_controller_state(BOOT_CONFIG.neutral_control)

    return backend.render(), {
        "boot_state": BootState.GP_RACE.value,
        "boot_last_phase": last_state.value,
        "frame_index": backend.frame_index,
        "reset_mode": "boot_to_race",
    }


def _press_start(backend: EmulatorBackend) -> None:
    backend.set_controller_state(BOOT_CONFIG.start_control)
    backend.step_frames(BOOT_CONFIG.start_hold_frames)
    backend.set_controller_state(BOOT_CONFIG.neutral_control)


def continue_to_next_race(backend: EmulatorBackend) -> tuple[RgbFrame, dict[str, object]]:
    """Advance the current session from a terminal state into the next race."""

    try:
        if _is_race_start_ready(backend):
            backend.step_frames(BOOT_CONFIG.pre_race_settle_frames)
            return _race_reset_result(backend, reset_mode="continue_to_next_race")

        for _ in range(BOOT_CONFIG.next_race_max_start_presses):
            _press_start(backend)
            for _ in range(BOOT_CONFIG.next_race_settle_frames):
                if _is_race_start_ready(backend):
                    backend.step_frames(BOOT_CONFIG.pre_race_settle_frames)
                    return _race_reset_result(
                        backend,
                        reset_mode="continue_to_next_race",
                    )
                _advance_poll_frame(backend)
    finally:
        backend.set_controller_state(BOOT_CONFIG.neutral_control)

    raise RuntimeError("Could not advance the current session into the next race")


def sync_race_intro_target(
    backend: EmulatorBackend,
    *,
    target_timer: int | None,
) -> tuple[dict[str, object], FZeroXTelemetry | None]:
    """Wait until the race countdown reaches the configured gameplay start point."""

    if target_timer is None:
        return {"race_intro_timer_sync": "skipped"}, _read_telemetry(backend)

    telemetry = _read_telemetry(backend)
    if telemetry is None:
        return {
            "race_intro_timer_target": target_timer,
            "race_intro_timer_sync": "skipped",
            "race_intro_timer_sync_reason": "telemetry_unavailable",
        }, None
    if not telemetry.in_race_mode:
        return {
            "race_intro_timer": telemetry.race_intro_timer,
            "race_intro_timer_target": target_timer,
            "race_intro_timer_sync": "skipped",
            "race_intro_timer_sync_reason": "not_in_race",
        }, telemetry
    if telemetry.race_intro_timer <= target_timer:
        return {
            "race_intro_timer": telemetry.race_intro_timer,
            "race_intro_timer_target": target_timer,
            "race_intro_timer_sync": "already_set",
            "race_intro_timer_waited_frames": 0,
        }, telemetry

    for waited_frames in range(1, BOOT_CONFIG.race_intro_wait_frames + 1):
        _advance_poll_frame(backend)
        telemetry = _read_telemetry(backend)
        if telemetry is None:
            continue
        if telemetry.in_race_mode and telemetry.race_intro_timer <= target_timer:
            return {
                "race_intro_timer": telemetry.race_intro_timer,
                "race_intro_timer_target": target_timer,
                "race_intro_timer_sync": "changed",
                "race_intro_timer_waited_frames": waited_frames,
            }, telemetry

    telemetry = _read_telemetry(backend)
    current_timer = None if telemetry is None else telemetry.race_intro_timer
    raise RuntimeError(
        "Timed out waiting for race intro timer "
        f"<= {target_timer}; current timer is {current_timer!r}."
    )


def _wait_for_gp_race(backend: EmulatorBackend) -> bool:
    return _wait_for_mode(
        backend,
        BOOT_CONFIG.race_mode_name,
        BOOT_CONFIG.race_mode_wait_frames,
    )


def _wait_for_mode(
    backend: EmulatorBackend,
    mode_name: str,
    max_frames: int,
) -> bool:
    for _ in range(max_frames):
        if _current_game_mode_name(backend) == mode_name:
            return True
        _advance_poll_frame(backend)
    return _current_game_mode_name(backend) == mode_name


def _is_race_start_ready(backend: EmulatorBackend) -> bool:
    telemetry = _read_telemetry(backend)
    if telemetry is None:
        return False
    return (
        telemetry.game_mode_name == BOOT_CONFIG.race_mode_name
        and telemetry.player.race_time_ms == 0
        and not telemetry.player.finished
        and not telemetry.player.crashed
        and not telemetry.player.retired
    )


def _read_telemetry(backend: EmulatorBackend):
    return backend.try_read_telemetry()


def _current_game_mode_name(backend: EmulatorBackend) -> str | None:
    telemetry = _read_telemetry(backend)
    if telemetry is None:
        return None
    return telemetry.game_mode_name


def _settle_after_input(backend: EmulatorBackend, max_frames: int) -> None:
    starting_mode_name = _current_game_mode_name(backend)
    if starting_mode_name is None:
        backend.step_frames(max_frames)
        return

    for _ in range(max_frames):
        current_mode_name = _current_game_mode_name(backend)
        if current_mode_name is not None and current_mode_name != starting_mode_name:
            return
        _advance_poll_frame(backend)


def _advance_poll_frame(backend: EmulatorBackend) -> None:
    """Advance one telemetry-poll frame without materializing a Python frame object."""

    backend.step_frames(1, capture_video=True)


def _race_reset_result(
    backend: EmulatorBackend,
    *,
    reset_mode: str,
) -> tuple[RgbFrame, dict[str, object]]:
    return backend.render(), {
        "boot_state": BootState.GP_RACE.value,
        "frame_index": backend.frame_index,
        "reset_mode": reset_mode,
    }
