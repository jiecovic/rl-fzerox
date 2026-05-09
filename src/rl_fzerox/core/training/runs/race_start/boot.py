# src/rl_fzerox/core/training/runs/race_start/boot.py
from __future__ import annotations

from fzerox_emulator import (
    JOYPAD_BUTTONS,
    ControllerState,
    Emulator,
    joypad_mask,
)
from rl_fzerox.core.boot import UNLOCK_EVERYTHING_SEQUENCE
from rl_fzerox.core.runtime_spec.vehicle_catalog import vehicle_menu_row_and_column
from rl_fzerox.core.training.runs.race_start.exact import write_engine_settings
from rl_fzerox.core.training.runs.race_start.models import MENU_TIMING, RaceStartVariant
from rl_fzerox.core.training.runs.race_start.validation import (
    validate_boot_materialized_setup,
    validate_variant,
)


def materialize_race_start_from_boot(
    *,
    emulator: Emulator,
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
    emulator: Emulator,
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
    emulator: Emulator,
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
    emulator: Emulator,
    variant: RaceStartVariant,
) -> None:
    """Navigate from a cold boot and save a clean Time Attack race-start setup."""

    if variant.mode != "time_attack":
        raise ValueError(
            f"Boot-menu race-start materialization supports time_attack only, got {variant.mode!r}"
        )
    validate_variant(variant)

    emulator.reset()
    _release_input(emulator)
    emulator.step_frames(MENU_TIMING.boot_frames, capture_video=False)

    _press_until_mode(emulator, target_mode="main_menu")
    emulator.step_frames(MENU_TIMING.menu_ready_frames, capture_video=False)
    _unlock_everything(emulator)
    emulator.step_frames(MENU_TIMING.post_unlock_settle_frames, capture_video=False)
    emulator.patch_time_attack_menu_mode()
    _press_until_mode(emulator, target_mode="course_select")
    emulator.step_frames(MENU_TIMING.menu_ready_frames, capture_video=False)

    _select_time_attack_course(emulator, variant.course_index)
    _wait_until_mode(emulator, target_mode="machine_select")
    emulator.step_frames(MENU_TIMING.menu_ready_frames, capture_video=False)

    _select_machine(emulator, variant)
    _wait_until_mode(emulator, target_mode="machine_settings")
    emulator.step_frames(MENU_TIMING.menu_ready_frames, capture_video=False)

    write_engine_settings(emulator, variant)
    _press_until_mode(emulator, target_mode="time_attack", require_race_mode=True)
    _step_until_ready_from_boot(emulator, variant)


def materialize_time_attack_menu_seed(
    *,
    emulator: Emulator,
) -> None:
    """Navigate once to the Time Attack course-select menu and stop there."""

    emulator.reset()
    _release_input(emulator)
    emulator.step_frames(MENU_TIMING.boot_frames, capture_video=False)
    _press_until_mode(emulator, target_mode="main_menu")
    emulator.step_frames(MENU_TIMING.menu_ready_frames, capture_video=False)
    _unlock_everything(emulator)
    emulator.step_frames(MENU_TIMING.post_unlock_settle_frames, capture_video=False)
    emulator.patch_time_attack_menu_mode()
    _press_until_mode(emulator, target_mode="course_select")
    emulator.step_frames(MENU_TIMING.menu_ready_frames, capture_video=False)


def materialize_time_attack_race_start_from_menu_seed(
    *,
    emulator: Emulator,
    variant: RaceStartVariant,
) -> None:
    """Navigate from a cached Time Attack course-select seed to the race start."""

    if variant.mode != "time_attack":
        raise ValueError(
            f"Menu-seed race-start materialization supports time_attack only, got {variant.mode!r}"
        )
    validate_variant(variant)

    emulator.reset()
    _release_input(emulator)
    _wait_until_mode(emulator, target_mode="course_select")
    emulator.step_frames(MENU_TIMING.menu_ready_frames, capture_video=False)

    _select_time_attack_course(emulator, variant.course_index)
    _wait_until_mode(emulator, target_mode="machine_select")
    emulator.step_frames(MENU_TIMING.menu_ready_frames, capture_video=False)

    _select_machine(emulator, variant)
    _wait_until_mode(emulator, target_mode="machine_settings")
    emulator.step_frames(MENU_TIMING.menu_ready_frames, capture_video=False)

    write_engine_settings(emulator, variant)
    _press_until_mode(emulator, target_mode="time_attack", require_race_mode=True)
    _step_until_ready_from_boot(emulator, variant)


def materialize_gp_race_start_from_boot(
    *,
    emulator: Emulator,
    variant: RaceStartVariant,
) -> None:
    """Navigate from a cold boot and save a clean GP race-start setup."""

    if variant.mode != "gp_race":
        raise ValueError(
            f"Boot-menu race-start materialization supports gp_race only, got {variant.mode!r}"
        )
    validate_variant(variant)

    emulator.reset()
    _release_input(emulator)
    emulator.step_frames(MENU_TIMING.boot_frames, capture_video=False)

    _press_until_mode(emulator, target_mode="main_menu")
    emulator.step_frames(MENU_TIMING.menu_ready_frames, capture_video=False)
    _unlock_everything(emulator)
    emulator.step_frames(MENU_TIMING.post_unlock_settle_frames, capture_video=False)
    _press_until_mode(emulator, target_mode="course_select")
    emulator.step_frames(MENU_TIMING.menu_ready_frames, capture_video=False)
    _press_until_mode(emulator, target_mode="machine_select")
    emulator.step_frames(MENU_TIMING.menu_ready_frames, capture_video=False)
    _apply_exact_race_start_setup(emulator, variant)
    _step_until_ready_from_boot(emulator, variant)


def materialize_gp_race_menu_seed(
    *,
    emulator: Emulator,
) -> None:
    """Navigate once to the GP machine-select menu and stop there."""

    emulator.reset()
    _release_input(emulator)
    emulator.step_frames(MENU_TIMING.boot_frames, capture_video=False)
    _press_until_mode(emulator, target_mode="main_menu")
    emulator.step_frames(MENU_TIMING.menu_ready_frames, capture_video=False)
    _unlock_everything(emulator)
    emulator.step_frames(MENU_TIMING.post_unlock_settle_frames, capture_video=False)
    _press_until_mode(emulator, target_mode="course_select")
    emulator.step_frames(MENU_TIMING.menu_ready_frames, capture_video=False)
    _press_until_mode(emulator, target_mode="machine_select")
    emulator.step_frames(MENU_TIMING.menu_ready_frames, capture_video=False)


def materialize_gp_race_start_from_menu_seed(
    *,
    emulator: Emulator,
    variant: RaceStartVariant,
) -> None:
    """Navigate from a cached GP machine-select seed to the race start."""

    if variant.mode != "gp_race":
        raise ValueError(
            f"Menu-seed race-start materialization supports gp_race only, got {variant.mode!r}"
        )
    validate_variant(variant)

    emulator.reset()
    _release_input(emulator)
    _wait_until_mode(emulator, target_mode="machine_select")
    emulator.step_frames(MENU_TIMING.menu_ready_frames, capture_video=False)
    _apply_exact_race_start_setup(emulator, variant)
    _step_until_ready_from_boot(emulator, variant)


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


def _select_time_attack_course(emulator: Emulator, course_index: int) -> None:
    if not 0 <= course_index < 24:
        raise ValueError(
            "Boot-menu materialization currently supports built-in cup courses 0..23, "
            f"got course_index={course_index}"
        )

    emulator.step_frames(MENU_TIMING.menu_ready_frames, capture_video=False)
    cup_index, track_index = divmod(course_index, 6)
    for _ in range(cup_index):
        _tap_menu_right(emulator)
    _tap_start(emulator)
    for _ in range(track_index):
        _tap_menu_right(emulator)
    _tap_start(emulator)
    _tap_start(emulator)


def _select_machine(emulator: Emulator, variant: RaceStartVariant) -> None:
    emulator.step_frames(MENU_TIMING.menu_ready_frames, capture_video=False)
    row, column = _machine_select_row_and_column(variant)
    for _ in range(row):
        _tap_menu_down(emulator)
    for _ in range(column):
        _tap_menu_right(emulator)
    _tap_start(emulator)
    _tap_start(emulator)


def _machine_select_row_and_column(variant: RaceStartVariant) -> tuple[int, int]:
    if variant.machine_select_slot is not None:
        return vehicle_menu_row_and_column(variant.machine_select_slot)
    return vehicle_menu_row_and_column(variant.character_index)


def _apply_exact_race_start_setup(emulator: Emulator, variant: RaceStartVariant) -> None:
    emulator.patch_machine_settings(
        mode=variant.mode,
        course_index=variant.course_index,
        character_index=variant.character_index,
        engine_setting_raw_value=variant.engine_setting_raw_value,
        total_lap_count=variant.total_lap_count,
        gp_difficulty=variant.gp_difficulty,
    )
    emulator.patch_race_start_setup(
        mode=variant.mode,
        course_index=variant.course_index,
        character_index=variant.character_index,
        engine_setting_raw_value=variant.engine_setting_raw_value,
        total_lap_count=variant.total_lap_count,
        gp_difficulty=variant.gp_difficulty,
    )
    emulator.force_race_reinit(mode=variant.mode)


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
        _tap_start(emulator)

    telemetry = emulator.try_read_telemetry()
    mode = None if telemetry is None else telemetry.game_mode_name
    raise RuntimeError(
        f"Boot-menu materialization did not reach {target_mode!r}; last mode was {mode!r}"
    )


def _wait_until_mode(
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
        emulator.step_frames(MENU_TIMING.menu_settle_frames, capture_video=False)

    telemetry = emulator.try_read_telemetry()
    mode = None if telemetry is None else telemetry.game_mode_name
    raise RuntimeError(
        f"Boot-menu materialization did not passively reach {target_mode!r}; last mode was {mode!r}"
    )


def _unlock_everything(emulator: Emulator) -> None:
    for unlock_input in UNLOCK_EVERYTHING_SEQUENCE:
        _tap_state(
            emulator,
            unlock_input.control_state,
            hold_frames=MENU_TIMING.menu_hold_frames,
            settle_frames=MENU_TIMING.menu_settle_frames,
        )


def _tap_start(emulator: Emulator, *, capture_video: bool = False) -> None:
    _tap_state(
        emulator,
        ControllerState(joypad_mask=joypad_mask(JOYPAD_BUTTONS.start)),
        hold_frames=MENU_TIMING.start_hold_frames,
        settle_frames=MENU_TIMING.start_settle_frames,
        capture_video=capture_video,
    )


def _tap_menu_right(emulator: Emulator, *, capture_video: bool = False) -> None:
    _tap_state(
        emulator,
        ControllerState(joypad_mask=joypad_mask(JOYPAD_BUTTONS.right)),
        hold_frames=MENU_TIMING.menu_hold_frames,
        settle_frames=MENU_TIMING.menu_settle_frames,
        capture_video=capture_video,
    )


def _tap_menu_down(emulator: Emulator, *, capture_video: bool = False) -> None:
    _tap_state(
        emulator,
        ControllerState(joypad_mask=joypad_mask(JOYPAD_BUTTONS.down)),
        hold_frames=MENU_TIMING.menu_hold_frames,
        settle_frames=MENU_TIMING.menu_settle_frames,
        capture_video=capture_video,
    )


def _tap_state(
    emulator: Emulator,
    control_state: ControllerState,
    *,
    hold_frames: int,
    settle_frames: int,
    capture_video: bool = False,
) -> None:
    emulator.set_controller_state(control_state)
    emulator.step_frames(hold_frames, capture_video=capture_video)
    _release_input(emulator)
    emulator.step_frames(settle_frames, capture_video=capture_video)


def _release_input(emulator: Emulator) -> None:
    emulator.set_controller_state(ControllerState())
