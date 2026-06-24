# src/rl_fzerox/core/training/runs/race_start/navigation.py
"""Shared boot-menu navigation primitives for race-start materialization."""

from __future__ import annotations

from fzerox_emulator import (
    JOYPAD_BUTTONS,
    ControllerState,
    EmulatorBackend,
    joypad_mask,
)
from rl_fzerox.core.boot import UNLOCK_EVERYTHING_SEQUENCE
from rl_fzerox.core.training.runs.race_start.menu_route import (
    machine_select_route_for_variant,
)
from rl_fzerox.core.training.runs.race_start.models import MENU_TIMING, RaceStartVariant
from rl_fzerox.core.training.runs.race_start.validation import (
    validate_boot_materialized_setup,
)


def step_until_ready_from_boot(emulator: EmulatorBackend, variant: RaceStartVariant) -> None:
    last_summary = "telemetry unavailable"
    saw_new_race_init = False
    target_timer = variant.race_intro_target_timer
    for _ in range(MENU_TIMING.race_init_frame_limit):
        emulator.step_frames(1, capture_video=True)
        telemetry = emulator.try_read_telemetry()
        if telemetry is None:
            continue

        last_summary = (
            f"mode={telemetry.game_mode_name!r}, course={telemetry.course_index}, "
            f"intro={telemetry.race_intro_timer}"
        )
        if not telemetry.in_race_mode:
            continue
        if telemetry.game_mode_name != variant.mode:
            continue
        if int(telemetry.course_index) != variant.course_index:
            continue
        if int(telemetry.race_intro_timer) > 100:
            saw_new_race_init = True
        if saw_new_race_init and (
            target_timer is None or int(telemetry.race_intro_timer) <= target_timer
        ):
            validate_boot_materialized_setup(emulator, variant)
            return

    raise RuntimeError(
        "Boot-menu baseline materialization did not reach the requested GO-window "
        f"state within {MENU_TIMING.race_init_frame_limit} frames ({last_summary})"
    )


def select_machine(emulator: EmulatorBackend, variant: RaceStartVariant) -> None:
    emulator.step_frames(MENU_TIMING.menu_ready_frames, capture_video=False)
    route = machine_select_route_for_variant(variant)
    for _ in range(route.down_count):
        tap_menu_down(emulator)
    for _ in range(route.right_count):
        tap_menu_right(emulator)
    tap_start(emulator)
    tap_start(emulator)


def press_until_mode(
    emulator: EmulatorBackend,
    *,
    target_mode: str,
    require_race_mode: bool = False,
) -> None:
    for _ in range(MENU_TIMING.mode_press_limit):
        telemetry = emulator.try_read_telemetry()
        if telemetry is not None and telemetry.game_mode_name == target_mode:
            if not require_race_mode or telemetry.in_race_mode:
                return
        tap_start(emulator)

    telemetry = emulator.try_read_telemetry()
    mode = None if telemetry is None else telemetry.game_mode_name
    raise RuntimeError(
        f"Boot-menu materialization did not reach {target_mode!r}; last mode was {mode!r}"
    )


def wait_until_mode(
    emulator: EmulatorBackend,
    *,
    target_mode: str,
    require_race_mode: bool = False,
) -> None:
    for _ in range(MENU_TIMING.mode_press_limit):
        telemetry = emulator.try_read_telemetry()
        if telemetry is not None and telemetry.game_mode_name == target_mode:
            if not require_race_mode or telemetry.in_race_mode:
                return
        emulator.step_frames(MENU_TIMING.menu_settle_frames, capture_video=False)

    telemetry = emulator.try_read_telemetry()
    mode = None if telemetry is None else telemetry.game_mode_name
    raise RuntimeError(
        f"Boot-menu materialization did not passively reach {target_mode!r}; last mode was {mode!r}"
    )


def unlock_everything(emulator: EmulatorBackend) -> None:
    for unlock_input in UNLOCK_EVERYTHING_SEQUENCE:
        tap_state(
            emulator,
            unlock_input.control_state,
            hold_frames=MENU_TIMING.menu_hold_frames,
            settle_frames=MENU_TIMING.menu_settle_frames,
        )


def tap_start(emulator: EmulatorBackend, *, capture_video: bool = False) -> None:
    tap_state(
        emulator,
        ControllerState(joypad_mask=joypad_mask(JOYPAD_BUTTONS.start)),
        hold_frames=MENU_TIMING.start_hold_frames,
        settle_frames=MENU_TIMING.start_settle_frames,
        capture_video=capture_video,
    )


def tap_menu_right(emulator: EmulatorBackend, *, capture_video: bool = False) -> None:
    tap_state(
        emulator,
        ControllerState(joypad_mask=joypad_mask(JOYPAD_BUTTONS.right)),
        hold_frames=MENU_TIMING.menu_hold_frames,
        settle_frames=MENU_TIMING.menu_settle_frames,
        capture_video=capture_video,
    )


def tap_menu_down(emulator: EmulatorBackend, *, capture_video: bool = False) -> None:
    tap_state(
        emulator,
        ControllerState(joypad_mask=joypad_mask(JOYPAD_BUTTONS.down)),
        hold_frames=MENU_TIMING.menu_hold_frames,
        settle_frames=MENU_TIMING.menu_settle_frames,
        capture_video=capture_video,
    )


def tap_state(
    emulator: EmulatorBackend,
    control_state: ControllerState,
    *,
    hold_frames: int,
    settle_frames: int,
    capture_video: bool = False,
) -> None:
    emulator.set_controller_state(control_state)
    emulator.step_frames(hold_frames, capture_video=capture_video)
    release_input(emulator)
    emulator.step_frames(settle_frames, capture_video=capture_video)


def release_input(emulator: EmulatorBackend) -> None:
    emulator.set_controller_state(ControllerState())
