# tests/core/manager/test_manager_store_save_games.py
from __future__ import annotations

from pathlib import Path
from typing import Literal

import pytest

from rl_fzerox.core.career_mode.progress import default_unlock_targets
from rl_fzerox.core.domain.courses import BUILT_IN_COURSES
from rl_fzerox.core.manager import (
    ManagerStore,
    default_managed_run_config,
)
from rl_fzerox.core.manager.db import manager_session
from rl_fzerox.core.manager.db.models import (
    SaveGameAttemptModel,
    SaveGameCourseSetupModel,
    SaveGameCupSetupModel,
    SaveGameModel,
)
from rl_fzerox.core.manager.errors import ManagerNameConflictError
from tests.core.manager.manager_store_support import (
    _logical_sra,
    _write_policy_artifact,
)

SnapshotKind = Literal["run", "draft", "template", "import"]


def test_manager_store_creates_save_game_record(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    save_games_root = tmp_path / "save_games"

    save_game = store.create_save_game(
        name="Unlock Run",
        save_games_root=save_games_root,
    )

    assert save_game.name == "Unlock Run"
    assert save_game.status == "created"
    assert save_game.save_path == save_games_root / save_game.id / "fzerox.srm"
    assert save_game.save_path.parent.is_dir()
    assert not save_game.save_path.exists()

    save_games = store.list_save_games()
    assert len(save_games) == 1
    assert save_games[0] == save_game
    progress = store.save_game_unlock_progress(save_game.id)
    assert progress.inspection_status == "not_inspected"
    assert progress.completed_count == 0
    assert progress.total_count == len(default_unlock_targets())
    assert progress.next_target is not None
    assert progress.next_target.kind == "clear_gp_cup"
    assert progress.next_target.difficulty == "novice"
    assert progress.next_target.cup_id == "jack"
    with manager_session(store.db_path) as session:
        assert session.get(SaveGameModel, save_game.id) is not None


def test_manager_store_rejects_duplicate_save_game_names_without_directory(
    tmp_path: Path,
) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    save_games_root = tmp_path / "save_games"
    store.create_save_game(name="Unlock Run", save_games_root=save_games_root)
    existing_dirs = {path.name for path in save_games_root.iterdir()}

    with pytest.raises(ManagerNameConflictError, match="name already exists"):
        store.create_save_game(name="unlock run", save_games_root=save_games_root)

    assert {path.name for path in save_games_root.iterdir()} == existing_dirs


def test_manager_store_deletes_save_game_with_owned_rows_and_files(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    save_game = store.create_save_game(
        name="Delete Save",
        save_games_root=tmp_path / "save-games",
    )
    save_game.save_path.write_bytes(_logical_sra({}))
    run = store.create_run(
        name="Policy Run",
        config=default_managed_run_config(),
        managed_runs_root=tmp_path / "runs",
    )
    course_setup = store.upsert_save_course_setup(
        save_game_id=save_game.id,
        cup_id="jack",
        course_id="mute_city",
        policy_run_id=run.id,
        policy_artifact="best",
    )
    cup_setup = store.upsert_save_cup_setup(
        save_game_id=save_game.id,
        cup_id="jack",
        vehicle_id="blue_falcon",
    )
    attempt = store.start_save_attempt(
        save_game_id=save_game.id,
        target_kind="clear_gp_cup",
        difficulty="novice",
        cup_id="jack",
    )

    assert store.delete_save_game(save_game.id) is True

    assert store.get_save_game(save_game.id) is None
    assert not save_game.save_path.parent.exists()
    with manager_session(store.db_path) as session:
        assert session.get(SaveGameModel, save_game.id) is None
        assert session.get(SaveGameCourseSetupModel, course_setup.id) is None
        assert session.get(SaveGameCupSetupModel, cup_setup.id) is None
        assert session.get(SaveGameAttemptModel, attempt.id) is None


def test_manager_store_rejects_delete_of_running_save_game(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    save_game = store.create_save_game(
        name="Running Save",
        save_games_root=tmp_path / "save-games",
    )
    updated = store.update_save_game_status(save_game_id=save_game.id, status="running")
    assert updated is not None

    with pytest.raises(ValueError, match="career runner"):
        store.delete_save_game(save_game.id)

    assert store.get_save_game(save_game.id) is not None


def test_manager_store_upserts_save_course_setups(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    run = store.create_run(
        run_id="policy-run",
        name="Policy Run",
        config=default_managed_run_config(),
        managed_runs_root=tmp_path / "runs",
    )
    save_game = store.create_save_game(
        name="Unlock Save",
        save_games_root=tmp_path / "save-games",
    )

    created = store.upsert_save_course_setup(
        save_game_id=save_game.id,
        cup_id="jack",
        course_id="mute_city",
        policy_run_id=run.id,
        policy_artifact="best",
    )
    updated = store.upsert_save_course_setup(
        save_game_id=save_game.id,
        cup_id="jack",
        course_id="mute_city",
        policy_run_id=run.id,
        policy_artifact="latest",
        engine_setting_raw_value=60,
    )

    assignments = store.list_save_course_setups(save_game.id)
    assert len(assignments) == 1
    assert created.id == updated.id
    assert assignments[0].save_game_id == save_game.id
    assert assignments[0].policy_run_id == "policy-run"
    assert assignments[0].policy_artifact == "latest"
    assert assignments[0].cup_id == "jack"
    assert assignments[0].course_id == "mute_city"
    assert assignments[0].engine_setting_raw_value == 60


def test_manager_store_records_save_target_attempts(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    save_game = store.create_save_game(
        name="Unlock Save",
        save_games_root=tmp_path / "save-games",
    )
    attempt = store.start_save_attempt(
        save_game_id=save_game.id,
        target_kind="clear_gp_cup",
        difficulty="novice",
        cup_id="jack",
    )
    finished = store.finish_save_attempt(
        attempt_id=attempt.id,
        status="succeeded",
        finish_position=1,
        finish_time_s=123.4,
    )

    attempts = store.list_save_attempts(save_game.id)
    assert len(attempts) == 1
    assert finished == attempts[0]
    assert attempts[0].save_game_id == save_game.id
    assert attempts[0].target_kind == "clear_gp_cup"
    assert attempts[0].status == "succeeded"
    assert attempts[0].finish_position == 1
    assert attempts[0].finish_time_s == 123.4
    assert attempts[0].finished_at is not None


def test_manager_store_starts_next_save_attempt_from_course_setup(
    tmp_path: Path,
) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    save_game = store.create_save_game(
        name="Policy Attempt Save",
        save_games_root=tmp_path / "save-games",
    )
    run = store.create_run(
        name="Unlock Policy",
        config=default_managed_run_config(),
        managed_runs_root=tmp_path / "runs",
    )
    _configure_gp_cup(store, save_game_id=save_game.id, run_id=run.id, cup_id="jack")

    attempt = store.start_next_save_attempt(save_game.id)

    assert attempt.save_game_id == save_game.id
    assert attempt.target_kind == "clear_gp_cup"
    assert attempt.difficulty == "novice"
    assert attempt.cup_id == "jack"
    assert attempt.status == "running"


def test_manager_store_starts_next_save_attempt_with_default_vehicle(
    tmp_path: Path,
) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    save_game = store.create_save_game(
        name="Default Vehicle Attempt Save",
        save_games_root=tmp_path / "save-games",
    )
    run = store.create_run(
        name="Unlock Policy",
        config=default_managed_run_config(),
        managed_runs_root=tmp_path / "runs",
    )
    _write_policy_artifact(run.run_dir, "best")
    _configure_gp_cup(
        store,
        save_game_id=save_game.id,
        run_id=run.id,
        cup_id="jack",
        include_cup_setup=False,
    )

    attempt = store.start_next_save_attempt(save_game.id)
    context = store.get_save_attempt_execution_context(attempt.id)

    assert attempt.status == "running"
    assert context is not None
    assert context.cup_setup.vehicle_id == "blue_falcon"


def test_manager_store_starts_selected_save_attempt_from_course_setup(
    tmp_path: Path,
) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    save_game = store.create_save_game(
        name="Selected Policy Attempt Save",
        save_games_root=tmp_path / "save-games",
    )
    save_game.save_path.write_bytes(_logical_sra({}))
    run = store.create_run(
        name="Selected Unlock Policy",
        config=default_managed_run_config(),
        managed_runs_root=tmp_path / "runs",
    )
    _configure_gp_cup(store, save_game_id=save_game.id, run_id=run.id, cup_id="king")

    attempt = store.start_target_save_attempt(
        save_game.id,
        target_kind="clear_gp_cup",
        difficulty="novice",
        cup_id="king",
    )

    assert attempt.save_game_id == save_game.id
    assert attempt.target_kind == "clear_gp_cup"
    assert attempt.difficulty == "novice"
    assert attempt.cup_id == "king"
    assert attempt.status == "running"


def test_manager_store_rejects_selected_save_attempt_before_save_inspection(
    tmp_path: Path,
) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    save_game = store.create_save_game(
        name="Uninspected Selected Policy Save",
        save_games_root=tmp_path / "save-games",
    )
    run = store.create_run(
        name="Selected Unlock Policy",
        config=default_managed_run_config(),
        managed_runs_root=tmp_path / "runs",
    )
    _configure_gp_cup(store, save_game_id=save_game.id, run_id=run.id, cup_id="king")

    with pytest.raises(ValueError, match="selected unlock target is locked"):
        store.start_target_save_attempt(
            save_game.id,
            target_kind="clear_gp_cup",
            difficulty="novice",
            cup_id="king",
        )


def test_manager_store_resolves_save_attempt_execution_context(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    save_game = store.create_save_game(
        name="Context Save",
        save_games_root=tmp_path / "save-games",
    )
    run = store.create_run(
        name="Context Policy",
        config=default_managed_run_config(),
        managed_runs_root=tmp_path / "runs",
    )
    policy_path = _write_policy_artifact(run.run_dir, "best")
    _configure_gp_cup(store, save_game_id=save_game.id, run_id=run.id, cup_id="jack")

    attempt = store.start_next_save_attempt(save_game.id)
    context = store.get_save_attempt_execution_context(attempt.id)

    assert context is not None
    assert context.save_game == save_game
    assert context.attempt == attempt
    assert context.target.kind == "clear_gp_cup"
    assert context.target.difficulty == "novice"
    assert context.target.cup_id == "jack"
    assert context.cup_setup.vehicle_id == "blue_falcon"
    assert context.policy_run.id == run.id
    assert context.policy_path == policy_path.resolve()


def _configure_gp_cup(
    store: ManagerStore,
    *,
    save_game_id: str,
    run_id: str,
    cup_id: str,
    include_cup_setup: bool = True,
) -> None:
    if include_cup_setup:
        store.upsert_save_cup_setup(
            save_game_id=save_game_id,
            cup_id=cup_id,
            vehicle_id="blue_falcon",
        )
    for course in sorted(BUILT_IN_COURSES, key=lambda item: item.course_index):
        if course.cup != cup_id:
            continue
        store.upsert_save_course_setup(
            save_game_id=save_game_id,
            cup_id=cup_id,
            course_id=course.id,
            policy_run_id=run_id,
            policy_artifact="best",
        )
