# src/rl_fzerox/core/training/runs/race_start/exact.py
"""Exact race-start capture loop for already selected course and vehicle setups."""

from __future__ import annotations

from fzerox_emulator import EmulatorBackend
from rl_fzerox.core.training.runs.race_start.boundary import (
    race_start_gp_difficulty_raw_value,
)
from rl_fzerox.core.training.runs.race_start.models import RACE_DEFAULTS, RaceStartVariant
from rl_fzerox.core.training.runs.race_start.validation import (
    validate_materialized_setup,
    validate_mode,
    validate_variant,
)


def materialize_race_start_state(
    *,
    emulator: EmulatorBackend,
    variant: RaceStartVariant,
) -> None:
    """Patch race setup globals and let the game rebuild a clean race start."""

    validate_mode(variant.mode)
    validate_variant(variant)

    emulator.reset()
    _write_full_machine_settings(emulator, variant)
    _write_race_setup(emulator, variant)
    _randomize_game_rng(emulator, variant)
    _force_race_reinit(emulator, variant)
    _step_until_ready(emulator, variant)


def write_engine_settings(emulator: EmulatorBackend, variant: RaceStartVariant) -> None:
    emulator.patch_engine_settings(
        mode=variant.mode,
        engine_setting_raw_value=variant.engine_setting_raw_value,
    )


def _write_race_setup(emulator: EmulatorBackend, variant: RaceStartVariant) -> None:
    emulator.patch_race_start_setup(
        mode=variant.mode,
        course_index=variant.course_index,
        character_index=variant.character_index,
        engine_setting_raw_value=variant.engine_setting_raw_value,
        total_lap_count=variant.total_lap_count,
        gp_difficulty_raw_value=race_start_gp_difficulty_raw_value(variant),
    )


def _write_full_machine_settings(emulator: EmulatorBackend, variant: RaceStartVariant) -> None:
    emulator.patch_machine_settings(
        mode=variant.mode,
        course_index=variant.course_index,
        character_index=variant.character_index,
        engine_setting_raw_value=variant.engine_setting_raw_value,
        total_lap_count=variant.total_lap_count,
        gp_difficulty_raw_value=race_start_gp_difficulty_raw_value(variant),
    )


def _randomize_game_rng(emulator: EmulatorBackend, variant: RaceStartVariant) -> None:
    if variant.rng_seed is None:
        return
    emulator.randomize_game_rng(variant.rng_seed)


def _force_race_reinit(emulator: EmulatorBackend, variant: RaceStartVariant) -> None:
    emulator.force_race_reinit(mode=variant.mode)


def _step_until_ready(emulator: EmulatorBackend, variant: RaceStartVariant) -> None:
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
        if telemetry.game_mode_name != variant.mode:
            continue
        if int(telemetry.course_index) != variant.course_index:
            continue
        if int(telemetry.race_intro_timer) > 100:
            saw_new_race_init = True
        if saw_new_race_init and (
            target_timer is None or int(telemetry.race_intro_timer) <= target_timer
        ):
            _write_race_setup(emulator, variant)
            validate_materialized_setup(emulator, variant)
            return

    raise RuntimeError(
        "Race-start baseline materialization did not reach the requested GO-window "
        f"state within {RACE_DEFAULTS.max_init_frames} frames ({last_summary})"
    )
