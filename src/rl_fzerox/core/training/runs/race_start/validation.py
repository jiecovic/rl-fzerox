# src/rl_fzerox/core/training/runs/race_start/validation.py
from __future__ import annotations

from fzerox_emulator import EmulatorBackend
from rl_fzerox.core.domain.engine_setting import validate_engine_slider_step
from rl_fzerox.core.domain.race_difficulty import race_difficulty_names
from rl_fzerox.core.training.runs.race_start.boundary import (
    race_start_gp_difficulty_raw_value,
)
from rl_fzerox.core.training.runs.race_start.models import RaceStartVariant


def validate_mode(mode: str) -> None:
    if mode in ("time_attack", "gp_race"):
        return
    raise ValueError(f"Unsupported race-start mode {mode!r}")


def validate_variant(variant: RaceStartVariant) -> None:
    validate_mode(variant.mode)
    if variant.mode == "time_attack" and variant.gp_difficulty is not None:
        raise ValueError("gp_difficulty is only supported for gp_race variants")
    if variant.mode == "gp_race" and (
        variant.gp_difficulty is not None and variant.gp_difficulty not in race_difficulty_names()
    ):
        raise ValueError(f"Unsupported gp_difficulty {variant.gp_difficulty!r}")
    if variant.course_index < 0:
        raise ValueError(f"course_index must be non-negative, got {variant.course_index}")
    if variant.character_index < 0:
        raise ValueError(f"character_index must be non-negative, got {variant.character_index}")
    if variant.machine_select_slot is not None and variant.machine_select_slot < 0:
        raise ValueError(
            f"machine_select_slot must be non-negative, got {variant.machine_select_slot}"
        )
    validate_engine_slider_step(variant.engine_setting_raw_value, label="engine_setting raw value")
    if variant.total_lap_count <= 0:
        raise ValueError(f"total_lap_count must be positive, got {variant.total_lap_count}")


def validate_materialized_setup(emulator: EmulatorBackend, variant: RaceStartVariant) -> None:
    emulator.validate_race_start_setup(
        mode=variant.mode,
        course_index=variant.course_index,
        character_index=variant.character_index,
        engine_setting_raw_value=variant.engine_setting_raw_value,
        total_lap_count=variant.total_lap_count,
        gp_difficulty_raw_value=race_start_gp_difficulty_raw_value(variant),
    )
    _validate_machine_identity(emulator, variant)


def validate_boot_materialized_setup(emulator: EmulatorBackend, variant: RaceStartVariant) -> None:
    mismatches: list[str] = []
    try:
        validate_materialized_setup(emulator, variant)
    except RuntimeError as error:
        mismatches.append(str(error))

    telemetry = emulator.try_read_telemetry()
    if telemetry is None:
        mismatches.append("telemetry: unavailable")
    else:
        if telemetry.game_mode_name != variant.mode:
            mismatches.append(
                f"game_mode: expected {variant.mode!r}, got {telemetry.game_mode_name!r}"
            )
        if int(telemetry.course_index) != variant.course_index:
            mismatches.append(
                f"telemetry.course_index: expected {variant.course_index}, "
                f"got {telemetry.course_index}"
            )
        if variant.mode == "gp_race" and variant.gp_difficulty is not None:
            if telemetry.difficulty_name != variant.gp_difficulty:
                mismatches.append(
                    "telemetry.difficulty_name: expected "
                    f"{variant.gp_difficulty!r}, got {telemetry.difficulty_name!r}"
                )

    if mismatches:
        raise RuntimeError(
            "Boot-menu baseline materialization produced inconsistent RAM state: "
            + "; ".join(mismatches)
        )


def _validate_machine_identity(emulator: EmulatorBackend, variant: RaceStartVariant) -> None:
    vehicle_info = emulator.vehicle_setup_info()
    raw_character_index = vehicle_info.get("racer_character_index_ram")
    if isinstance(raw_character_index, bool) or not isinstance(raw_character_index, int):
        return
    if int(raw_character_index) != variant.character_index:
        raise RuntimeError(
            "vehicle_setup_info mismatch: expected "
            f"character_index={variant.character_index}, got {raw_character_index}"
        )
