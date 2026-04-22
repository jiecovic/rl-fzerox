# src/rl_fzerox/core/training/runs/baseline_race_start.py
from __future__ import annotations

from dataclasses import dataclass

from fzerox_emulator import (
    JOYPAD_RIGHT,
    JOYPAD_START,
    ControllerState,
    Emulator,
    joypad_mask,
)


@dataclass(frozen=True, slots=True)
class RaceStartDefaults:
    """Race-start defaults used when the caller does not override a variant field."""

    lap_count: int = 3
    machine_skin_index: int = 0
    max_init_frames: int = 720


@dataclass(frozen=True, slots=True)
class RaceStartMenuTiming:
    """Frame timing used by the cold-boot menu materializer."""

    boot_frames: int = 300
    menu_ready_frames: int = 90
    button_hold_frames: int = 2
    button_settle_frames: int = 38
    mode_press_limit: int = 24
    race_init_frame_limit: int = 1_500


RACE_DEFAULTS = RaceStartDefaults()
MENU_TIMING = RaceStartMenuTiming()


@dataclass(frozen=True, slots=True)
class RaceStartVariant:
    """Target race-start setup materialized from an existing baseline state."""

    course_index: int
    mode: str
    character_index: int
    engine_setting_raw_value: int
    race_intro_target_timer: int | None
    machine_select_slot: int | None = None
    machine_skin_index: int = RACE_DEFAULTS.machine_skin_index
    total_lap_count: int = RACE_DEFAULTS.lap_count


def materialize_race_start_state(
    *,
    emulator: Emulator,
    variant: RaceStartVariant,
) -> None:
    """Patch race setup globals and let the game rebuild a clean race start."""

    if variant.mode != "time_attack":
        raise ValueError(
            f"Race-start materialization currently supports time_attack only, got {variant.mode!r}"
        )
    _validate_variant(variant)

    emulator.reset()
    _write_race_setup(emulator, variant)
    _force_time_attack_reinit(emulator)
    _step_until_ready(emulator, variant)


def materialize_time_attack_race_start_from_boot(
    *,
    emulator: Emulator,
    variant: RaceStartVariant,
) -> None:
    """Navigate from a cold boot and save a clean Time Attack race-start setup.

    Vehicle and engine are selected before the race is initialized. This avoids
    corrupting an already-built live racer by patching machine identity after
    models/state have been constructed.
    """

    if variant.mode != "time_attack":
        raise ValueError(
            f"Boot-menu race-start materialization supports time_attack only, got {variant.mode!r}"
        )
    _validate_variant(variant)

    emulator.reset()
    _release_input(emulator)
    emulator.step_frames(MENU_TIMING.boot_frames, capture_video=False)

    _press_until_mode(emulator, target_mode="main_menu")
    emulator.step_frames(MENU_TIMING.menu_ready_frames, capture_video=False)
    _write_machine_settings(emulator, variant)
    _press_until_mode(emulator, target_mode="course_select")

    _select_time_attack_course(emulator, variant.course_index)
    _press_until_mode(emulator, target_mode="machine_select")

    _select_machine(emulator, variant)
    _press_until_mode(emulator, target_mode="machine_settings")

    _write_machine_settings(emulator, variant)
    _press_until_mode(emulator, target_mode="time_attack", require_race_mode=True)
    _step_until_ready_from_boot(emulator, variant)


def _validate_variant(variant: RaceStartVariant) -> None:
    if variant.course_index < 0:
        raise ValueError(f"course_index must be non-negative, got {variant.course_index}")
    if variant.character_index < 0:
        raise ValueError(f"character_index must be non-negative, got {variant.character_index}")
    if variant.machine_select_slot is not None and variant.machine_select_slot < 0:
        raise ValueError(
            f"machine_select_slot must be non-negative, got {variant.machine_select_slot}"
        )
    if variant.machine_skin_index < 0:
        raise ValueError(
            f"machine_skin_index must be non-negative, got {variant.machine_skin_index}"
        )
    if not 0 <= variant.engine_setting_raw_value <= 100:
        raise ValueError(
            f"engine_setting raw value must be in [0, 100], got {variant.engine_setting_raw_value}"
        )
    if variant.total_lap_count <= 0:
        raise ValueError(f"total_lap_count must be positive, got {variant.total_lap_count}")


def _write_race_setup(emulator: Emulator, variant: RaceStartVariant) -> None:
    emulator.patch_time_attack_race_start_setup(
        course_index=variant.course_index,
        character_index=variant.character_index,
        machine_skin_index=variant.machine_skin_index,
        engine_setting_raw_value=variant.engine_setting_raw_value,
        total_lap_count=variant.total_lap_count,
    )


def _write_machine_settings(emulator: Emulator, variant: RaceStartVariant) -> None:
    emulator.patch_time_attack_machine_settings(
        course_index=variant.course_index,
        character_index=variant.character_index,
        machine_skin_index=variant.machine_skin_index,
        engine_setting_raw_value=variant.engine_setting_raw_value,
        total_lap_count=variant.total_lap_count,
    )


def _force_time_attack_reinit(emulator: Emulator) -> None:
    emulator.force_time_attack_reinit()


def _step_until_ready(emulator: Emulator, variant: RaceStartVariant) -> None:
    last_summary = "telemetry unavailable"
    saw_new_race_init = False
    target_timer = variant.race_intro_target_timer
    for _ in range(RACE_DEFAULTS.max_init_frames):
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
        if telemetry.game_mode_name != "time_attack":
            continue
        if int(telemetry.course_index) != variant.course_index:
            continue
        if int(telemetry.race_intro_timer) > 100:
            saw_new_race_init = True
        if saw_new_race_init and (
            target_timer is None or int(telemetry.race_intro_timer) <= target_timer
        ):
            _write_race_setup(emulator, variant)
            _validate_materialized_setup(emulator, variant)
            return

    raise RuntimeError(
        "Race-start baseline materialization did not reach the requested GO-window "
        f"state within {RACE_DEFAULTS.max_init_frames} frames ({last_summary})"
    )


def _step_until_ready_from_boot(emulator: Emulator, variant: RaceStartVariant) -> None:
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
        if telemetry.game_mode_name != "time_attack":
            continue
        if int(telemetry.course_index) != variant.course_index:
            continue
        if int(telemetry.race_intro_timer) > 100:
            saw_new_race_init = True
        if saw_new_race_init and (
            target_timer is None or int(telemetry.race_intro_timer) <= target_timer
        ):
            _validate_boot_materialized_setup(emulator, variant)
            return

    raise RuntimeError(
        "Boot-menu baseline materialization did not reach the requested GO-window "
        f"state within {MENU_TIMING.race_init_frame_limit} frames ({last_summary})"
    )


def _select_time_attack_course(emulator: Emulator, course_index: int) -> None:
    if not 0 <= course_index < 24:
        raise ValueError(
            "Boot-menu materialization currently supports built-in cup courses 0..23, "
            f"got course_index={course_index}"
        )

    emulator.step_frames(MENU_TIMING.menu_ready_frames, capture_video=False)
    cup_index, track_index = divmod(course_index, 6)
    for _ in range(cup_index):
        _tap_button(emulator, JOYPAD_RIGHT)
    _tap_button(emulator, JOYPAD_START)
    for _ in range(track_index):
        _tap_button(emulator, JOYPAD_RIGHT)
    _tap_button(emulator, JOYPAD_START)
    _tap_button(emulator, JOYPAD_START)


def _select_machine(emulator: Emulator, variant: RaceStartVariant) -> None:
    emulator.step_frames(MENU_TIMING.menu_ready_frames, capture_video=False)
    for _ in range(_machine_select_right_presses(variant)):
        _tap_button(emulator, JOYPAD_RIGHT)
    _tap_button(emulator, JOYPAD_START)


def _machine_select_right_presses(variant: RaceStartVariant) -> int:
    # The machine-select menu order is not the same as internal character ids
    # for the full roster. Stock vehicle metadata supplies the menu slot.
    if variant.machine_select_slot is not None:
        return variant.machine_select_slot
    return variant.character_index


def _press_until_mode(
    emulator: Emulator,
    *,
    target_mode: str,
    require_race_mode: bool = False,
) -> None:
    for _ in range(MENU_TIMING.mode_press_limit):
        telemetry = emulator.try_read_telemetry()
        if telemetry is not None and telemetry.game_mode_name == target_mode:
            if not require_race_mode or telemetry.in_race_mode:
                return
        _tap_button(emulator, JOYPAD_START)

    telemetry = emulator.try_read_telemetry()
    mode = None if telemetry is None else telemetry.game_mode_name
    raise RuntimeError(
        f"Boot-menu materialization did not reach {target_mode!r}; last mode was {mode!r}"
    )


def _tap_button(emulator: Emulator, button: int, *, capture_video: bool = False) -> None:
    emulator.set_controller_state(ControllerState(joypad_mask=joypad_mask(button)))
    emulator.step_frames(MENU_TIMING.button_hold_frames, capture_video=capture_video)
    _release_input(emulator)
    emulator.step_frames(MENU_TIMING.button_settle_frames, capture_video=capture_video)


def _release_input(emulator: Emulator) -> None:
    emulator.set_controller_state(ControllerState())


def _validate_materialized_setup(emulator: Emulator, variant: RaceStartVariant) -> None:
    emulator.validate_time_attack_race_start_setup(
        course_index=variant.course_index,
        character_index=variant.character_index,
        machine_skin_index=variant.machine_skin_index,
        engine_setting_raw_value=variant.engine_setting_raw_value,
        total_lap_count=variant.total_lap_count,
    )


def _validate_boot_materialized_setup(emulator: Emulator, variant: RaceStartVariant) -> None:
    mismatches: list[str] = []
    try:
        _validate_materialized_setup(emulator, variant)
    except RuntimeError as error:
        mismatches.append(str(error))

    telemetry = emulator.try_read_telemetry()
    if telemetry is None:
        mismatches.append("telemetry: unavailable")
    else:
        if telemetry.game_mode_name != "time_attack":
            mismatches.append(
                f"game_mode: expected 'time_attack', got {telemetry.game_mode_name!r}"
            )
        if int(telemetry.course_index) != variant.course_index:
            mismatches.append(
                f"telemetry.course_index: expected {variant.course_index}, "
                f"got {telemetry.course_index}"
            )

    if mismatches:
        raise RuntimeError(
            "Boot-menu baseline materialization produced inconsistent RAM state: "
            + "; ".join(mismatches)
        )
