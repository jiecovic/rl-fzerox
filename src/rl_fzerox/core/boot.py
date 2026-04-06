# src/rl_fzerox/core/boot.py
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from rl_fzerox._native import JOYPAD_START, joypad_mask
from rl_fzerox.core.emulator.base import EmulatorBackend

START_MASK = joypad_mask(JOYPAD_START)
TITLE_WAIT_FRAMES = 240
START_HOLD_FRAMES = 2
SETTLE_FRAMES = 60
RACE_GRID_WAIT_FRAMES = 360


class BootState(StrEnum):
    """Named phases for the deterministic first-race bootstrap path."""

    WAIT_FOR_TITLE = "wait_for_title"
    OPEN_MODE_MENU = "open_mode_menu"
    CONFIRM_DIFFICULTY = "confirm_difficulty"
    CONFIRM_COURSE = "confirm_course"
    ENTER_MACHINE_SELECT = "enter_machine_select"
    CONFIRM_MACHINE = "confirm_machine"
    CONFIRM_MACHINE_PROFILE = "confirm_machine_profile"
    WAIT_FOR_GRID = "wait_for_grid"
    READY = "ready"


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
    BootPhase(BootState.WAIT_FOR_GRID, start_presses=0, settle_frames=RACE_GRID_WAIT_FRAMES),
)


def boot_into_first_race(backend: EmulatorBackend) -> tuple[np.ndarray, dict[str, object]]:
    """Drive the default USA boot path into the first Mute City grid.

    This is intentionally timing-based for now. Once we expose enough game
    state from RAM, the next step is to validate these phase transitions
    against real menu/race state instead of relying on fixed frame counts.
    """

    last_state = BootState.READY
    try:
        for phase in BOOT_SEQUENCE:
            for _ in range(phase.start_presses):
                _press_start(backend)
                backend.step_frames(phase.settle_frames)
            if phase.start_presses == 0:
                backend.step_frames(phase.settle_frames)
            last_state = phase.state
    finally:
        backend.set_joypad_mask(0)

    return backend.render(), {
        "boot_state": BootState.READY.value,
        "boot_last_phase": last_state.value,
        "frame_index": backend.frame_index,
        "reset_mode": "boot_to_race",
    }


def _press_start(backend: EmulatorBackend) -> None:
    backend.set_joypad_mask(START_MASK)
    backend.step_frames(START_HOLD_FRAMES)
    backend.set_joypad_mask(0)
