# src/rl_fzerox/core/boot.py
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import cast

import numpy as np

from rl_fzerox._native import JOYPAD_START, joypad_mask
from rl_fzerox.core.emulator.base import EmulatorBackend
from rl_fzerox.core.game.telemetry import GameMode, MemoryReadableEmulator, read_telemetry

START_MASK = joypad_mask(JOYPAD_START)
TITLE_WAIT_FRAMES = 240
START_HOLD_FRAMES = 2
SETTLE_FRAMES = 60
RACE_MODE_WAIT_FRAMES = 360
PRE_RACE_SETTLE_FRAMES = 180
NEXT_RACE_MAX_START_PRESSES = 12
NEXT_RACE_SETTLE_FRAMES = 45


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
    settle_frames: int = SETTLE_FRAMES


# This sequence was measured against the current USA ROM boot path with the
# default GP Race / Novice / Jack Cup / Blue Falcon selections.
BOOT_SEQUENCE: tuple[BootPhase, ...] = (
    BootPhase(BootState.WAIT_FOR_TITLE, start_presses=0, settle_frames=TITLE_WAIT_FRAMES),
    BootPhase(BootState.OPEN_MODE_MENU, start_presses=4),
    BootPhase(BootState.CONFIRM_DIFFICULTY, start_presses=2),
    BootPhase(BootState.CONFIRM_COURSE, start_presses=2),
    BootPhase(BootState.ENTER_MACHINE_SELECT, start_presses=2),
    BootPhase(BootState.CONFIRM_MACHINE, start_presses=2),
    BootPhase(BootState.CONFIRM_MACHINE_PROFILE, start_presses=4),
)


def boot_into_first_race(backend: EmulatorBackend) -> tuple[np.ndarray, dict[str, object]]:
    """Drive the default USA boot path into the first Mute City grid.

    This is intentionally timing-based for now. Once we expose enough game
    state from RAM, the next step is to validate these phase transitions
    against real menu/race state instead of relying on fixed frame counts.
    """

    last_state = BootState.GP_RACE
    try:
        for phase in BOOT_SEQUENCE:
            for _ in range(phase.start_presses):
                _press_start(backend)
                backend.step_frames(phase.settle_frames)
            if phase.start_presses == 0:
                backend.step_frames(phase.settle_frames)
            last_state = phase.state
        detected_gp_race = _wait_for_gp_race(backend)
        last_state = BootState.WAIT_FOR_GP_RACE
        if detected_gp_race:
            backend.step_frames(PRE_RACE_SETTLE_FRAMES)
    finally:
        backend.set_joypad_mask(0)

    return backend.render(), {
        "boot_state": BootState.GP_RACE.value,
        "boot_last_phase": last_state.value,
        "frame_index": backend.frame_index,
        "reset_mode": "boot_to_race",
    }


def _press_start(backend: EmulatorBackend) -> None:
    backend.set_joypad_mask(START_MASK)
    backend.step_frames(START_HOLD_FRAMES)
    backend.set_joypad_mask(0)


def continue_to_next_race(backend: EmulatorBackend) -> tuple[np.ndarray, dict[str, object]]:
    """Advance the current session from a terminal state into the next race."""

    try:
        if _is_race_start_ready(backend):
            backend.step_frames(PRE_RACE_SETTLE_FRAMES)
            return _race_reset_result(backend, reset_mode="continue_to_next_race")

        for _ in range(NEXT_RACE_MAX_START_PRESSES):
            _press_start(backend)
            for _ in range(NEXT_RACE_SETTLE_FRAMES):
                if _is_race_start_ready(backend):
                    backend.step_frames(PRE_RACE_SETTLE_FRAMES)
                    return _race_reset_result(
                        backend,
                        reset_mode="continue_to_next_race",
                    )
                backend.step_frame()
    finally:
        backend.set_joypad_mask(0)

    raise RuntimeError("Could not advance the current session into the next race")


def _wait_for_gp_race(backend: EmulatorBackend) -> bool:
    for _ in range(RACE_MODE_WAIT_FRAMES):
        if _is_gp_race(backend):
            return True
        backend.step_frame()
    return False


def _is_gp_race(backend: EmulatorBackend) -> bool:
    telemetry = _read_telemetry(backend)
    if telemetry is None:
        return False
    return (telemetry.game_mode_raw & 0x1F) == int(GameMode.GP_RACE)


def _is_race_start_ready(backend: EmulatorBackend) -> bool:
    telemetry = _read_telemetry(backend)
    if telemetry is None:
        return False
    return (
        (telemetry.game_mode_raw & 0x1F) == int(GameMode.GP_RACE)
        and telemetry.player.race_time_ms == 0
        and "finished" not in telemetry.player.state_labels
        and "crashed" not in telemetry.player.state_labels
        and "retired" not in telemetry.player.state_labels
    )


def _read_telemetry(backend: EmulatorBackend):
    if not hasattr(backend, "system_ram_size") or not hasattr(backend, "read_system_ram"):
        return None
    try:
        return read_telemetry(cast(MemoryReadableEmulator, backend))
    except (AttributeError, RuntimeError):
        return None


def _race_reset_result(
    backend: EmulatorBackend,
    *,
    reset_mode: str,
) -> tuple[np.ndarray, dict[str, object]]:
    return backend.render(), {
        "boot_state": BootState.GP_RACE.value,
        "frame_index": backend.frame_index,
        "reset_mode": reset_mode,
    }
