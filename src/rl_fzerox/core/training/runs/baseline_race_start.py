# src/rl_fzerox/core/training/runs/baseline_race_start.py
from __future__ import annotations

import struct
from dataclasses import dataclass

from fzerox_emulator import (
    JOYPAD_RIGHT,
    JOYPAD_START,
    ControllerState,
    Emulator,
    joypad_mask,
)


@dataclass(frozen=True, slots=True)
class RaceStartRamLayout:
    """RAM offsets touched when preparing a race-start save state."""

    num_players: int = 0x000C_D000
    total_lap_count: int = 0x000C_D00C
    selected_mode: int = 0x000C_D380
    game_mode_change_state: int = 0x000C_D046
    current_ghost_type: int = 0x000C_D3CC
    game_mode: int = 0x000D_CE44
    queued_game_mode: int = 0x000D_CE48
    character_last_engine: int = 0x000E_40F0
    player_characters: int = 0x000E_5EE0
    player_machine_skins: int = 0x000E_5EE8
    player_engine: int = 0x000E_5EF0
    course_index: int = 0x000F_8514
    player_racer_base: int = 0x002C_4920
    racer_engine_curve: int = 0x1A8
    racer_character: int = 0x2C8
    racer_machine_skin_index: int = 0x2CC


@dataclass(frozen=True, slots=True)
class RaceStartGameIds:
    """Game enum values needed by the RAM-backed race-start materializer."""

    time_attack_game_mode: int = 0x0E
    change_init: int = 3
    time_attack_menu_mode: int = 1
    no_ghost: int = 0
    single_player: int = 1


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


RAM_LAYOUT = RaceStartRamLayout()
GAME_IDS = RaceStartGameIds()
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
    _write_i32(emulator, RAM_LAYOUT.selected_mode, GAME_IDS.time_attack_menu_mode)
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
    engine_value = variant.engine_setting_raw_value / 100.0
    engine_curve = _engine_to_curve_value(engine_value)

    _write_i32(emulator, RAM_LAYOUT.num_players, GAME_IDS.single_player)
    _write_i32(emulator, RAM_LAYOUT.total_lap_count, variant.total_lap_count)
    _write_i32(emulator, RAM_LAYOUT.selected_mode, GAME_IDS.time_attack_menu_mode)
    _write_i32(emulator, RAM_LAYOUT.current_ghost_type, GAME_IDS.no_ghost)
    _write_i32(emulator, RAM_LAYOUT.course_index, variant.course_index)

    _write_i16(emulator, RAM_LAYOUT.player_characters, variant.character_index)
    _write_i16(emulator, RAM_LAYOUT.player_machine_skins, variant.machine_skin_index)
    _write_f32(emulator, RAM_LAYOUT.player_engine, engine_value)
    _write_f32(
        emulator,
        RAM_LAYOUT.character_last_engine + (variant.character_index * 4),
        engine_value,
    )

    _write_i8(
        emulator,
        RAM_LAYOUT.player_racer_base + RAM_LAYOUT.racer_character,
        variant.character_index,
    )
    _write_i16(
        emulator,
        RAM_LAYOUT.player_racer_base + RAM_LAYOUT.racer_machine_skin_index,
        variant.machine_skin_index,
    )
    _write_f32(
        emulator,
        RAM_LAYOUT.player_racer_base + RAM_LAYOUT.racer_engine_curve,
        engine_curve,
    )


def _write_machine_settings(emulator: Emulator, variant: RaceStartVariant) -> None:
    engine_value = variant.engine_setting_raw_value / 100.0
    _write_i32(emulator, RAM_LAYOUT.num_players, GAME_IDS.single_player)
    _write_i32(emulator, RAM_LAYOUT.total_lap_count, variant.total_lap_count)
    _write_i32(emulator, RAM_LAYOUT.selected_mode, GAME_IDS.time_attack_menu_mode)
    _write_i32(emulator, RAM_LAYOUT.current_ghost_type, GAME_IDS.no_ghost)
    _write_i32(emulator, RAM_LAYOUT.course_index, variant.course_index)
    _write_i16(emulator, RAM_LAYOUT.player_characters, variant.character_index)
    _write_i16(emulator, RAM_LAYOUT.player_machine_skins, variant.machine_skin_index)
    _write_f32(emulator, RAM_LAYOUT.player_engine, engine_value)
    _write_f32(
        emulator,
        RAM_LAYOUT.character_last_engine + (variant.character_index * 4),
        engine_value,
    )


def _force_time_attack_reinit(emulator: Emulator) -> None:
    _write_i32(emulator, RAM_LAYOUT.game_mode, GAME_IDS.time_attack_game_mode)
    _write_i32(emulator, RAM_LAYOUT.queued_game_mode, GAME_IDS.time_attack_game_mode)
    _write_i16(emulator, RAM_LAYOUT.game_mode_change_state, GAME_IDS.change_init)


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


def _engine_to_curve_value(engine_value: float) -> float:
    if engine_value == 0.0:
        return 0.0
    return 1.0 / (((1.0 + 0.6899998) / engine_value) - 0.6899998)


def _validate_materialized_setup(emulator: Emulator, variant: RaceStartVariant) -> None:
    engine_value = variant.engine_setting_raw_value / 100.0
    engine_curve = _engine_to_curve_value(engine_value)
    mismatches: list[str] = []
    for label, actual, expected in (
        ("course_index", _read_i32(emulator, RAM_LAYOUT.course_index), variant.course_index),
        (
            "selected_mode",
            _read_i32(emulator, RAM_LAYOUT.selected_mode),
            GAME_IDS.time_attack_menu_mode,
        ),
        (
            "current_ghost_type",
            _read_i32(emulator, RAM_LAYOUT.current_ghost_type),
            GAME_IDS.no_ghost,
        ),
        (
            "player_character",
            _read_i16(emulator, RAM_LAYOUT.player_characters),
            variant.character_index,
        ),
        (
            "racer_character",
            _read_i8(emulator, RAM_LAYOUT.player_racer_base + RAM_LAYOUT.racer_character),
            variant.character_index,
        ),
        (
            "racer_machine_skin",
            _read_i16(
                emulator,
                RAM_LAYOUT.player_racer_base + RAM_LAYOUT.racer_machine_skin_index,
            ),
            variant.machine_skin_index,
        ),
    ):
        if actual != expected:
            mismatches.append(f"{label}: expected {expected!r}, got {actual!r}")

    actual_engine = _read_f32(emulator, RAM_LAYOUT.player_engine)
    if abs(actual_engine - engine_value) > 0.001:
        mismatches.append(f"player_engine: expected {engine_value:.3f}, got {actual_engine:.3f}")
    actual_engine_curve = _read_f32(
        emulator,
        RAM_LAYOUT.player_racer_base + RAM_LAYOUT.racer_engine_curve,
    )
    if abs(actual_engine_curve - engine_curve) > 0.001:
        mismatches.append(
            f"racer_engine_curve: expected {engine_curve:.3f}, got {actual_engine_curve:.3f}"
        )

    if mismatches:
        raise RuntimeError(
            "Race-start baseline materialization produced inconsistent RAM state: "
            + "; ".join(mismatches)
        )


def _validate_boot_materialized_setup(emulator: Emulator, variant: RaceStartVariant) -> None:
    engine_value = variant.engine_setting_raw_value / 100.0
    engine_curve = _engine_to_curve_value(engine_value)
    mismatches: list[str] = []
    for label, actual, expected in (
        ("course_index", _read_i32(emulator, RAM_LAYOUT.course_index), variant.course_index),
        (
            "selected_mode",
            _read_i32(emulator, RAM_LAYOUT.selected_mode),
            GAME_IDS.time_attack_menu_mode,
        ),
        (
            "current_ghost_type",
            _read_i32(emulator, RAM_LAYOUT.current_ghost_type),
            GAME_IDS.no_ghost,
        ),
        (
            "player_character",
            _read_i16(emulator, RAM_LAYOUT.player_characters),
            variant.character_index,
        ),
        (
            "racer_character",
            _read_i8(emulator, RAM_LAYOUT.player_racer_base + RAM_LAYOUT.racer_character),
            variant.character_index,
        ),
        (
            "racer_machine_skin",
            _read_i16(
                emulator,
                RAM_LAYOUT.player_racer_base + RAM_LAYOUT.racer_machine_skin_index,
            ),
            variant.machine_skin_index,
        ),
    ):
        if actual != expected:
            mismatches.append(f"{label}: expected {expected!r}, got {actual!r}")

    actual_engine = _read_f32(emulator, RAM_LAYOUT.player_engine)
    if abs(actual_engine - engine_value) > 0.001:
        mismatches.append(f"player_engine: expected {engine_value:.3f}, got {actual_engine:.3f}")
    actual_engine_curve = _read_f32(
        emulator,
        RAM_LAYOUT.player_racer_base + RAM_LAYOUT.racer_engine_curve,
    )
    if abs(actual_engine_curve - engine_curve) > 0.001:
        mismatches.append(
            f"racer_engine_curve: expected {engine_curve:.3f}, got {actual_engine_curve:.3f}"
        )

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


def _read_i8(emulator: Emulator, offset: int) -> int:
    return int(struct.unpack("<b", emulator.read_system_ram(offset, 1))[0])


def _read_i16(emulator: Emulator, offset: int) -> int:
    return int(struct.unpack("<h", emulator.read_system_ram(offset, 2))[0])


def _read_i32(emulator: Emulator, offset: int) -> int:
    return int(struct.unpack("<i", emulator.read_system_ram(offset, 4))[0])


def _read_f32(emulator: Emulator, offset: int) -> float:
    return float(struct.unpack("<f", emulator.read_system_ram(offset, 4))[0])


def _write_i8(emulator: Emulator, offset: int, value: int) -> None:
    emulator.write_system_ram(offset, struct.pack("<b", int(value)))


def _write_i16(emulator: Emulator, offset: int, value: int) -> None:
    emulator.write_system_ram(offset, struct.pack("<h", int(value)))


def _write_i32(emulator: Emulator, offset: int, value: int) -> None:
    emulator.write_system_ram(offset, struct.pack("<i", int(value)))


def _write_f32(emulator: Emulator, offset: int, value: float) -> None:
    emulator.write_system_ram(offset, struct.pack("<f", float(value)))
