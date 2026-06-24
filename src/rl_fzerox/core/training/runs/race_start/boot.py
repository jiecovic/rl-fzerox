# src/rl_fzerox/core/training/runs/race_start/boot.py
from __future__ import annotations

from fzerox_emulator import EmulatorBackend
from rl_fzerox.core.training.runs.race_start.boundary import (
    race_start_gp_difficulty_raw_value,
)
from rl_fzerox.core.training.runs.race_start.exact import write_engine_settings
from rl_fzerox.core.training.runs.race_start.models import MENU_TIMING, RaceStartVariant
from rl_fzerox.core.training.runs.race_start.navigation import (
    press_until_mode,
    release_input,
    select_machine,
    step_until_ready_from_boot,
    tap_menu_right,
    tap_start,
    unlock_everything,
    wait_until_mode,
)
from rl_fzerox.core.training.runs.race_start.validation import validate_variant


def materialize_race_start_from_boot(
    *,
    emulator: EmulatorBackend,
    variant: RaceStartVariant,
) -> None:
    """Navigate from a cold boot and save a clean race-start setup."""

    if variant.mode == "time_attack":
        materialize_time_attack_race_start_from_boot(emulator=emulator, variant=variant)
        return
    if variant.mode == "gp_race":
        materialize_gp_race_start_from_boot(emulator=emulator, variant=variant)
        return
    raise ValueError(f"Unsupported race-start mode {variant.mode!r}")


def materialize_generic_mode_seed(
    *,
    emulator: EmulatorBackend,
    mode: str,
) -> None:
    """Boot once into the stable menu state used to derive shared cache entries."""

    if mode == "time_attack":
        materialize_time_attack_menu_seed(emulator=emulator)
        return
    if mode == "gp_race":
        materialize_gp_race_menu_seed(emulator=emulator)
        return
    raise ValueError(f"Unsupported generic menu-seed mode {mode!r}")


def materialize_race_start_from_menu_seed(
    *,
    emulator: EmulatorBackend,
    variant: RaceStartVariant,
) -> None:
    """Derive one exact race-start baseline from a cached generic menu seed."""

    if variant.mode == "time_attack":
        materialize_time_attack_race_start_from_menu_seed(emulator=emulator, variant=variant)
        return
    if variant.mode == "gp_race":
        materialize_gp_race_start_from_menu_seed(emulator=emulator, variant=variant)
        return
    raise ValueError(f"Unsupported race-start mode {variant.mode!r}")


def materialize_time_attack_race_start_from_boot(
    *,
    emulator: EmulatorBackend,
    variant: RaceStartVariant,
) -> None:
    """Navigate from a cold boot and save a clean Time Attack race-start setup."""

    if variant.mode != "time_attack":
        raise ValueError(
            f"Boot-menu race-start materialization supports time_attack only, got {variant.mode!r}"
        )
    validate_variant(variant)

    emulator.reset()
    release_input(emulator)
    emulator.step_frames(MENU_TIMING.boot_frames, capture_video=False)

    press_until_mode(emulator, target_mode="main_menu")
    emulator.step_frames(MENU_TIMING.menu_ready_frames, capture_video=False)
    unlock_everything(emulator)
    emulator.step_frames(MENU_TIMING.post_unlock_settle_frames, capture_video=False)
    emulator.patch_time_attack_menu_mode()
    press_until_mode(emulator, target_mode="course_select")
    emulator.step_frames(MENU_TIMING.menu_ready_frames, capture_video=False)

    _select_time_attack_course(emulator, variant.course_index)
    wait_until_mode(emulator, target_mode="machine_select")
    emulator.step_frames(MENU_TIMING.menu_ready_frames, capture_video=False)

    select_machine(emulator, variant)
    wait_until_mode(emulator, target_mode="machine_settings")
    emulator.step_frames(MENU_TIMING.menu_ready_frames, capture_video=False)

    write_engine_settings(emulator, variant)
    press_until_mode(emulator, target_mode="time_attack", require_race_mode=True)
    step_until_ready_from_boot(emulator, variant)


def materialize_time_attack_menu_seed(
    *,
    emulator: EmulatorBackend,
) -> None:
    """Navigate once to the Time Attack course-select menu and stop there."""

    emulator.reset()
    release_input(emulator)
    emulator.step_frames(MENU_TIMING.boot_frames, capture_video=False)
    press_until_mode(emulator, target_mode="main_menu")
    emulator.step_frames(MENU_TIMING.menu_ready_frames, capture_video=False)
    unlock_everything(emulator)
    emulator.step_frames(MENU_TIMING.post_unlock_settle_frames, capture_video=False)
    emulator.patch_time_attack_menu_mode()
    press_until_mode(emulator, target_mode="course_select")
    emulator.step_frames(MENU_TIMING.menu_ready_frames, capture_video=False)


def materialize_time_attack_race_start_from_menu_seed(
    *,
    emulator: EmulatorBackend,
    variant: RaceStartVariant,
) -> None:
    """Navigate from a cached Time Attack course-select seed to the race start."""

    if variant.mode != "time_attack":
        raise ValueError(
            f"Menu-seed race-start materialization supports time_attack only, got {variant.mode!r}"
        )
    validate_variant(variant)

    emulator.reset()
    release_input(emulator)
    wait_until_mode(emulator, target_mode="course_select")
    emulator.step_frames(MENU_TIMING.menu_ready_frames, capture_video=False)

    _select_time_attack_course(emulator, variant.course_index)
    wait_until_mode(emulator, target_mode="machine_select")
    emulator.step_frames(MENU_TIMING.menu_ready_frames, capture_video=False)

    select_machine(emulator, variant)
    wait_until_mode(emulator, target_mode="machine_settings")
    emulator.step_frames(MENU_TIMING.menu_ready_frames, capture_video=False)

    write_engine_settings(emulator, variant)
    press_until_mode(emulator, target_mode="time_attack", require_race_mode=True)
    step_until_ready_from_boot(emulator, variant)


def materialize_gp_race_start_from_boot(
    *,
    emulator: EmulatorBackend,
    variant: RaceStartVariant,
) -> None:
    """Navigate from a cold boot and save a clean GP race-start setup."""

    if variant.mode != "gp_race":
        raise ValueError(
            f"Boot-menu race-start materialization supports gp_race only, got {variant.mode!r}"
        )
    validate_variant(variant)

    emulator.reset()
    release_input(emulator)
    emulator.step_frames(MENU_TIMING.boot_frames, capture_video=False)

    press_until_mode(emulator, target_mode="main_menu")
    emulator.step_frames(MENU_TIMING.menu_ready_frames, capture_video=False)
    unlock_everything(emulator)
    emulator.step_frames(MENU_TIMING.post_unlock_settle_frames, capture_video=False)
    press_until_mode(emulator, target_mode="course_select")
    emulator.step_frames(MENU_TIMING.menu_ready_frames, capture_video=False)
    press_until_mode(emulator, target_mode="machine_select")
    emulator.step_frames(MENU_TIMING.menu_ready_frames, capture_video=False)

    select_machine(emulator, variant)
    wait_until_mode(emulator, target_mode="machine_settings")
    emulator.step_frames(MENU_TIMING.menu_ready_frames, capture_video=False)

    _apply_exact_race_start_setup(emulator, variant)
    step_until_ready_from_boot(emulator, variant)


def materialize_gp_race_menu_seed(
    *,
    emulator: EmulatorBackend,
) -> None:
    """Navigate once to the GP machine-select menu and stop there."""

    emulator.reset()
    release_input(emulator)
    emulator.step_frames(MENU_TIMING.boot_frames, capture_video=False)
    press_until_mode(emulator, target_mode="main_menu")
    emulator.step_frames(MENU_TIMING.menu_ready_frames, capture_video=False)
    unlock_everything(emulator)
    emulator.step_frames(MENU_TIMING.post_unlock_settle_frames, capture_video=False)
    press_until_mode(emulator, target_mode="course_select")
    emulator.step_frames(MENU_TIMING.menu_ready_frames, capture_video=False)
    press_until_mode(emulator, target_mode="machine_select")
    emulator.step_frames(MENU_TIMING.menu_ready_frames, capture_video=False)


def materialize_gp_race_start_from_menu_seed(
    *,
    emulator: EmulatorBackend,
    variant: RaceStartVariant,
) -> None:
    """Navigate from a cached GP machine-select seed to the race start."""

    if variant.mode != "gp_race":
        raise ValueError(
            f"Menu-seed race-start materialization supports gp_race only, got {variant.mode!r}"
        )
    validate_variant(variant)

    emulator.reset()
    release_input(emulator)
    wait_until_mode(emulator, target_mode="machine_select")
    emulator.step_frames(MENU_TIMING.menu_ready_frames, capture_video=False)

    select_machine(emulator, variant)
    wait_until_mode(emulator, target_mode="machine_settings")
    emulator.step_frames(MENU_TIMING.menu_ready_frames, capture_video=False)

    _apply_exact_race_start_setup(emulator, variant)
    step_until_ready_from_boot(emulator, variant)


def _select_time_attack_course(emulator: EmulatorBackend, course_index: int) -> None:
    if not 0 <= course_index < 24:
        raise ValueError(
            "Boot-menu materialization currently supports built-in cup courses 0..23, "
            f"got course_index={course_index}"
        )

    emulator.step_frames(MENU_TIMING.menu_ready_frames, capture_video=False)
    cup_index, track_index = divmod(course_index, 6)
    for _ in range(cup_index):
        tap_menu_right(emulator)
    tap_start(emulator)
    for _ in range(track_index):
        tap_menu_right(emulator)
    tap_start(emulator)
    tap_start(emulator)


def _apply_exact_race_start_setup(emulator: EmulatorBackend, variant: RaceStartVariant) -> None:
    emulator.patch_machine_settings(
        mode=variant.mode,
        course_index=variant.course_index,
        character_index=variant.character_index,
        engine_setting_raw_value=variant.engine_setting_raw_value,
        total_lap_count=variant.total_lap_count,
        gp_difficulty_raw_value=race_start_gp_difficulty_raw_value(variant),
    )
    emulator.patch_race_start_setup(
        mode=variant.mode,
        course_index=variant.course_index,
        character_index=variant.character_index,
        engine_setting_raw_value=variant.engine_setting_raw_value,
        total_lap_count=variant.total_lap_count,
        gp_difficulty_raw_value=race_start_gp_difficulty_raw_value(variant),
    )
    if variant.rng_seed is not None:
        emulator.randomize_game_rng(variant.rng_seed)
    emulator.force_race_reinit(mode=variant.mode)
