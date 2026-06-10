# src/rl_fzerox/apps/career_mode_cli/config.py
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from rl_fzerox.apps._cli import normalize_cli_overrides
from rl_fzerox.core.career_mode.runner.race import (
    SaveRaceSetup,
    build_save_race_execution_plan,
)
from rl_fzerox.core.manager import ManagerStore
from rl_fzerox.core.manager.projection.assembly import emulator_data
from rl_fzerox.core.runtime_spec.schema import (
    CareerModeRaceSetupConfig,
    EmulatorConfig,
    WatchAppConfig,
    WatchConfig,
)
from rl_fzerox.core.runtime_spec.watch_overrides import (
    apply_watch_config_delta,
    watch_config_delta_from_dotlist,
)


def resolve_career_mode_config(
    *,
    db_path: Path,
    attempt_seed: int | None,
    deterministic_policy: bool,
    save_attempt_id: str | None,
    save_game_id: str,
    overrides: Sequence[str],
) -> WatchAppConfig:
    store = ManagerStore(db_path)
    if save_attempt_id is None:
        attempt = store.start_or_reuse_next_save_attempt(save_game_id)
        save_attempt_id = attempt.id
    context = store.get_save_attempt_execution_context(save_attempt_id)
    if context is None:
        raise RuntimeError(f"save attempt disappeared before launch: {save_attempt_id}")
    if context.save_game.id != save_game_id:
        raise ValueError("save attempt does not belong to the requested save game")
    plan = build_save_race_execution_plan(context)
    config = career_mode_base_config(
        db_path=db_path,
        save_game_id=context.save_game.id,
        save_path=context.save_game.save_path,
        attempt_id=context.attempt.id,
        emulator=EmulatorConfig.model_validate(emulator_data(context.policy_run.config)),
        attempt_seed=attempt_seed,
        deterministic_policy=deterministic_policy,
        race_setup=career_mode_race_setup_config(plan.race_setup),
        label=context.target.label,
    )
    if delta := watch_config_delta_from_dotlist(normalize_cli_overrides(overrides)):
        config = apply_watch_config_delta(config, delta)

    store.update_save_game_status(save_game_id=save_game_id, status="running")
    return config


def career_mode_base_config(
    *,
    db_path: Path,
    save_game_id: str,
    save_path: Path,
    attempt_id: str,
    emulator: EmulatorConfig,
    attempt_seed: int | None,
    deterministic_policy: bool,
    race_setup: CareerModeRaceSetupConfig,
    label: str,
) -> WatchAppConfig:
    emulator = career_mode_emulator_config(
        emulator=emulator,
        save_path=save_path,
    )
    return WatchAppConfig(
        seed=None,
        emulator=emulator,
        policy=None,
        train=None,
        watch=WatchConfig(
            control_fps="auto",
            manager_db_path=db_path,
            managed_save_game_id=save_game_id,
            save_attempt_id=attempt_id,
            unlock_target_label=label,
            attempt_seed=attempt_seed,
            deterministic_policy=deterministic_policy,
            start_manual_control=False,
            career_mode_race_setup=race_setup,
        ),
    )


def career_mode_emulator_config(
    *,
    emulator: EmulatorConfig,
    save_path: Path,
) -> EmulatorConfig:
    runtime_dir = save_path.parent / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    return emulator.model_copy(
        update={
            "runtime_dir": runtime_dir,
            "baseline_state_path": None,
        },
    )


def career_mode_race_setup_config(race_setup: SaveRaceSetup) -> CareerModeRaceSetupConfig:
    return CareerModeRaceSetupConfig(
        difficulty=race_setup.difficulty,
        cup_id=race_setup.cup_id,
        course_id=race_setup.course_id,
        vehicle_id=race_setup.vehicle_id,
        vehicle_display_name=race_setup.vehicle_display_name,
        character_index=race_setup.character_index,
        machine_select_slot=race_setup.machine_select_slot,
        machine_select_row=race_setup.machine_select_row,
        machine_select_column=race_setup.machine_select_column,
        engine_setting_raw_value=race_setup.engine_setting_raw_value,
    )
