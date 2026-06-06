# tests/core/career_mode/runner/test_race.py

from __future__ import annotations

from pathlib import Path

import pytest

from rl_fzerox.core.career_mode.course_setup import CourseSetupTarget
from rl_fzerox.core.career_mode.progress import default_unlock_targets
from rl_fzerox.core.career_mode.runner.context import SaveAttemptExecutionContext
from rl_fzerox.core.career_mode.runner.race import (
    build_save_race_execution_plan,
)
from rl_fzerox.core.manager.models import (
    ManagedRun,
    ManagedSaveAttempt,
    ManagedSaveGame,
)
from rl_fzerox.core.manager.projection.assembly import effective_train_algorithm
from rl_fzerox.core.manager.run_spec import ManagedRunConfig, default_managed_run_config


def test_build_save_race_execution_plan_resolves_policy_vehicle_setup() -> None:
    base_config = default_managed_run_config()
    config = base_config.model_copy(
        update={
            "vehicle": base_config.vehicle.model_copy(
                update={
                    "selected_vehicle_ids": ("fire_stingray",),
                    "engine_mode": "fixed",
                    "engine_setting_raw_value": 100,
                }
            )
        }
    )
    context = _context(config=config)

    plan = build_save_race_execution_plan(context)

    assert plan.attempt_id == "attempt"
    assert plan.policy_run_id == "policy-run"
    assert plan.policy_run_dir == Path("/tmp/run")
    assert plan.policy_artifact == "best"
    assert plan.policy_algorithm == effective_train_algorithm(config)
    assert plan.policy_path == Path("/tmp/policy.zip")
    assert plan.race_setup.difficulty == "expert"
    assert plan.race_setup.cup_id == "jack"
    assert plan.race_setup.vehicle_id == "fire_stingray"
    assert plan.race_setup.vehicle_display_name == "Fire Stingray"
    assert plan.race_setup.character_index == 3
    assert plan.race_setup.machine_select_slot == 3
    assert plan.race_setup.machine_select_row == 0
    assert plan.race_setup.machine_select_column == 3
    assert plan.race_setup.engine_setting_id == "max_speed"
    assert plan.race_setup.engine_setting_raw_value == 100


def test_build_save_race_execution_plan_accepts_degenerate_engine_range() -> None:
    base_config = default_managed_run_config()
    config = base_config.model_copy(
        update={
            "vehicle": base_config.vehicle.model_copy(
                update={
                    "engine_mode": "random_range",
                    "engine_setting_min_raw_value": 70,
                    "engine_setting_max_raw_value": 70,
                }
            )
        }
    )

    plan = build_save_race_execution_plan(_context(config=config))

    assert plan.race_setup.engine_setting_id == "engine_70"
    assert plan.race_setup.engine_setting_raw_value == 70


def test_build_save_race_execution_plan_rejects_multiple_selected_vehicles() -> None:
    base_config = default_managed_run_config()
    config = base_config.model_copy(
        update={
            "vehicle": base_config.vehicle.model_copy(
                update={"selected_vehicle_ids": ("blue_falcon", "fire_stingray")}
            )
        }
    )

    with pytest.raises(ValueError, match="exactly one selected vehicle"):
        build_save_race_execution_plan(_context(config=config))


def test_build_save_race_execution_plan_rejects_engine_range() -> None:
    base_config = default_managed_run_config()
    config = base_config.model_copy(
        update={
            "vehicle": base_config.vehicle.model_copy(
                update={
                    "engine_mode": "random_range",
                    "engine_setting_min_raw_value": 20,
                    "engine_setting_max_raw_value": 80,
                }
            )
        }
    )

    with pytest.raises(ValueError, match="deterministic engine setting"):
        build_save_race_execution_plan(_context(config=config))


def _context(*, config: ManagedRunConfig | None = None) -> SaveAttemptExecutionContext:
    resolved_config = config if config is not None else default_managed_run_config()
    rule_target = next(
        target
        for target in default_unlock_targets()
        if target.difficulty == "expert" and target.cup_id == "jack"
    )
    target = rule_target.to_progress_target(status="pending")
    return SaveAttemptExecutionContext(
        save_game=ManagedSaveGame(
            id="save",
            name="Save",
            status="created",
            save_path=Path("/tmp/fzerox.srm"),
            created_at="2026-06-01T00:00:00+00:00",
            updated_at="2026-06-01T00:00:00+00:00",
        ),
        attempt=ManagedSaveAttempt(
            id="attempt",
            save_game_id="save",
            status="running",
            started_at="2026-06-01T00:00:00+00:00",
            policy_run_id="policy-run",
            policy_artifact="best",
            target_kind=target.kind,
            difficulty="expert",
            cup_id="jack",
        ),
        target=target,
        course_setup_target=CourseSetupTarget(difficulty="expert", cup_id="jack"),
        policy_run=ManagedRun(
            id="policy-run",
            name="Policy Run",
            status="stopped",
            config=resolved_config,
            config_hash="hash",
            run_dir=Path("/tmp/run"),
            created_at="2026-06-01T00:00:00+00:00",
            lineage_id="policy-run",
        ),
        policy_artifact="best",
        policy_path=Path("/tmp/policy.zip"),
    )
