# src/rl_fzerox/core/career_mode/runner/race.py
"""Race setup resolution for save-game unlock attempts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from rl_fzerox.core.career_mode.runner.context import SaveAttemptExecutionContext
from rl_fzerox.core.manager.projection.assembly import effective_train_algorithm
from rl_fzerox.core.manager.run_spec import ManagedRunConfig
from rl_fzerox.core.runtime_spec.vehicle_catalog import (
    EngineSetting,
    vehicle_by_id,
    vehicle_menu_row_and_column,
)
from rl_fzerox.core.runtime_spec.vehicle_catalog import (
    resolve_engine_setting as resolve_catalog_engine_setting,
)


@dataclass(frozen=True, slots=True)
class SaveRaceSetup:
    """Concrete GP setup the menu runner must select before policy handoff."""

    difficulty: str
    cup_id: str
    course_id: str | None
    vehicle_id: str
    vehicle_display_name: str
    character_index: int
    machine_select_slot: int
    machine_select_row: int
    machine_select_column: int
    engine_setting_id: str
    engine_setting_raw_value: int


@dataclass(frozen=True, slots=True)
class SaveRaceExecutionPlan:
    """Resolved inputs for launching one policy-backed GP race attempt."""

    attempt_id: str
    policy_run_id: str
    policy_run_dir: Path
    policy_artifact: Literal["latest", "best"]
    policy_algorithm: str
    policy_path: Path
    race_setup: SaveRaceSetup


def build_save_race_execution_plan(
    context: SaveAttemptExecutionContext,
) -> SaveRaceExecutionPlan:
    """Resolve manager state into the fixed setup needed by the race executor."""

    if context.target.kind != "clear_gp_cup":
        raise ValueError(f"unsupported unlock target kind: {context.target.kind}")
    if context.course_setup_target.difficulty is None or context.course_setup_target.cup_id is None:
        raise ValueError("save-game unlock attempt requires difficulty and cup")

    vehicle_id = _single_vehicle_id(context.policy_run.config)
    vehicle = vehicle_by_id(vehicle_id)
    row, column = vehicle_menu_row_and_column(vehicle.machine_select_slot)
    engine_setting = _deterministic_engine_setting(context.policy_run.config)
    return SaveRaceExecutionPlan(
        attempt_id=context.attempt.id,
        policy_run_id=context.policy_run.id,
        policy_run_dir=context.policy_run.run_dir,
        policy_artifact=context.policy_artifact,
        policy_algorithm=effective_train_algorithm(context.policy_run.config),
        policy_path=context.policy_path,
        race_setup=SaveRaceSetup(
            difficulty=context.course_setup_target.difficulty,
            cup_id=context.course_setup_target.cup_id,
            course_id=context.course_setup_target.course_id,
            vehicle_id=vehicle.id,
            vehicle_display_name=vehicle.display_name,
            character_index=vehicle.character_index,
            machine_select_slot=vehicle.machine_select_slot,
            machine_select_row=row,
            machine_select_column=column,
            engine_setting_id=engine_setting.id,
            engine_setting_raw_value=engine_setting.raw_value,
        ),
    )


def _single_vehicle_id(config: ManagedRunConfig) -> str:
    selected_vehicle_ids = config.vehicle.selected_vehicle_ids
    if len(selected_vehicle_ids) != 1:
        raise ValueError("save-game unlock attempts require exactly one selected vehicle")
    return selected_vehicle_ids[0]


def _deterministic_engine_setting(config: ManagedRunConfig) -> EngineSetting:
    vehicle_config = config.vehicle
    if vehicle_config.engine_mode == "fixed":
        raw_value = vehicle_config.engine_setting_raw_value
    elif vehicle_config.engine_setting_min_raw_value == vehicle_config.engine_setting_max_raw_value:
        raw_value = vehicle_config.engine_setting_min_raw_value
    else:
        raise ValueError("save-game unlock attempts require a deterministic engine setting")
    return resolve_catalog_engine_setting(
        raw_value,
        context="save-game unlock vehicle setup",
    )
