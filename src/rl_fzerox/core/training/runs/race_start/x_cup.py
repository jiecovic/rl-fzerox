# src/rl_fzerox/core/training/runs/race_start/x_cup.py
"""Boot-menu materialization for generated GP X Cup race starts."""

from __future__ import annotations

from dataclasses import dataclass

from fzerox_emulator import EmulatorBackend, FZeroXTelemetry
from rl_fzerox.core.domain.x_cup import X_CUP_COURSE
from rl_fzerox.core.training.runs.race_start.boot import (
    _press_until_mode,
    _release_input,
    _select_machine,
    _step_until_ready_from_boot,
    _tap_menu_right,
    _unlock_everything,
    _wait_until_mode,
)
from rl_fzerox.core.training.runs.race_start.boundary import (
    race_start_gp_difficulty_raw_value,
)
from rl_fzerox.core.training.runs.race_start.models import MENU_TIMING, RaceStartVariant
from rl_fzerox.core.training.runs.race_start.validation import validate_variant


@dataclass(frozen=True, slots=True)
class XCupMaterializedCourse:
    """Metadata read from the generated X Cup race after boot materialization."""

    segment_count: int
    course_length: float


def materialize_x_cup_race_start_from_boot(
    *,
    emulator: EmulatorBackend,
    variant: RaceStartVariant,
    rng_seed: int,
) -> XCupMaterializedCourse:
    """Generate one X Cup track by menu selection and deterministic RNG patching."""

    if variant.mode != X_CUP_COURSE.race_mode:
        raise ValueError(
            f"X Cup materialization supports {X_CUP_COURSE.race_mode} only, got {variant.mode!r}"
        )
    if variant.course_index != X_CUP_COURSE.course_index:
        raise ValueError(
            "X Cup materialization requires course_index="
            f"{X_CUP_COURSE.course_index}, got {variant.course_index}"
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
    _select_x_cup(emulator)
    emulator.randomize_game_rng(rng_seed)

    _press_until_mode(emulator, target_mode="machine_select")
    emulator.step_frames(MENU_TIMING.menu_ready_frames, capture_video=False)
    _select_machine(emulator, variant)
    _wait_until_mode(emulator, target_mode="machine_settings")
    emulator.step_frames(MENU_TIMING.menu_ready_frames, capture_video=False)
    _write_x_cup_machine_setup(emulator, variant)

    _press_until_mode(
        emulator,
        target_mode=X_CUP_COURSE.race_mode,
        require_race_mode=True,
    )
    _step_until_ready_from_boot(emulator, variant)
    return _materialized_course(emulator)


def _select_x_cup(emulator: EmulatorBackend) -> None:
    for _ in range(X_CUP_COURSE.menu_right_presses_from_jack):
        _tap_menu_right(emulator)
    telemetry = emulator.try_read_telemetry()
    if not _is_x_cup_course_selection(telemetry):
        course_index = None if telemetry is None else telemetry.course_index
        raise RuntimeError(
            "X Cup materialization did not land on the expected course slot; "
            f"expected course_index={X_CUP_COURSE.course_index}, got {course_index}"
        )


def _write_x_cup_machine_setup(emulator: EmulatorBackend, variant: RaceStartVariant) -> None:
    emulator.patch_machine_settings(
        mode=variant.mode,
        course_index=variant.course_index,
        character_index=variant.character_index,
        engine_setting_raw_value=variant.engine_setting_raw_value,
        total_lap_count=variant.total_lap_count,
        gp_difficulty_raw_value=race_start_gp_difficulty_raw_value(variant),
    )


def _materialized_course(emulator: EmulatorBackend) -> XCupMaterializedCourse:
    telemetry = emulator.try_read_telemetry()
    if telemetry is None:
        raise RuntimeError("X Cup materialization finished without telemetry")
    if not _is_x_cup_race(telemetry):
        raise RuntimeError(
            "X Cup materialization finished in an unexpected state: "
            f"mode={telemetry.game_mode_name!r}, course={telemetry.course_index}, "
            f"in_race={telemetry.in_race_mode}"
        )
    return XCupMaterializedCourse(
        segment_count=int(telemetry.course_segment_count),
        course_length=float(telemetry.course_length),
    )


def _is_x_cup_course_selection(telemetry: FZeroXTelemetry | None) -> bool:
    return bool(
        telemetry is not None
        and telemetry.game_mode_name == "course_select"
        and int(telemetry.course_index) == X_CUP_COURSE.course_index
    )


def _is_x_cup_race(telemetry: FZeroXTelemetry | None) -> bool:
    return bool(
        telemetry is not None
        and telemetry.in_race_mode
        and telemetry.game_mode_name == X_CUP_COURSE.race_mode
        and int(telemetry.course_index) == X_CUP_COURSE.course_index
    )
