# src/rl_fzerox/core/career_mode/execution/race.py
"""Race setup resolution for save-game unlock attempts.

This module validates that a save attempt has the policy, vehicle, course, and
engine settings needed to enter one GP race/cup through the Career Mode menu.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from rl_fzerox.core.career_mode.execution.context import SaveAttemptExecutionContext
from rl_fzerox.core.manager.models import PolicySourceKind
from rl_fzerox.core.manager.projection.assembly import effective_train_algorithm
from rl_fzerox.core.runtime_spec.vehicle_catalog import (
    vehicle_by_id,
)
from rl_fzerox.core.training.runs.race_start.menu_route import machine_select_route_for_slot


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
    engine_setting_raw_value: int


@dataclass(frozen=True, slots=True)
class SaveRaceExecutionPlan:
    """Resolved inputs for launching one policy-backed GP race attempt."""

    attempt_id: str
    policy_source_kind: PolicySourceKind
    policy_source_id: str
    policy_source_name: str
    policy_source_dir: Path
    policy_artifact: Literal["latest", "best", "final"]
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

    vehicle_id = context.cup_setup.vehicle_id
    vehicle = vehicle_by_id(vehicle_id)
    route = machine_select_route_for_slot(vehicle.machine_select_slot)
    return SaveRaceExecutionPlan(
        attempt_id=context.attempt.id,
        policy_source_kind=context.policy_source.kind,
        policy_source_id=context.policy_source.id,
        policy_source_name=context.policy_source.name,
        policy_source_dir=context.policy_source.source_dir,
        policy_artifact=context.policy_artifact,
        policy_algorithm=effective_train_algorithm(context.policy_source.config),
        policy_path=context.policy_path,
        race_setup=SaveRaceSetup(
            difficulty=context.course_setup_target.difficulty,
            cup_id=context.course_setup_target.cup_id,
            course_id=context.course_setup_target.course_id,
            vehicle_id=vehicle.id,
            vehicle_display_name=vehicle.display_name,
            character_index=vehicle.character_index,
            machine_select_slot=vehicle.machine_select_slot,
            machine_select_row=route.down_count,
            machine_select_column=route.right_count,
            engine_setting_raw_value=context.course_setup.engine_setting_raw_value,
        ),
    )
