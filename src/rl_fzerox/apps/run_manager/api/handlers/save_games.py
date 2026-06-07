# src/rl_fzerox/apps/run_manager/api/handlers/save_games.py
from __future__ import annotations

from fastapi import HTTPException

from rl_fzerox.apps.run_manager.api.contracts import (
    CreateSaveGameRequest,
    RunLauncher,
    StartCareerModeRequest,
    UpdateSaveGameRequest,
    UpsertSaveCourseSetupRequest,
)
from rl_fzerox.apps.run_manager.api.payloads import (
    save_attempt_execution_context_payload,
    save_game_payload,
)
from rl_fzerox.apps.run_manager.desktop import open_directory
from rl_fzerox.apps.run_manager.launching.save_games import active_career_mode_runner_pid
from rl_fzerox.core.career_mode.runner.race import build_save_race_execution_plan
from rl_fzerox.core.career_mode.runner.reports import save_race_execution_plan_report
from rl_fzerox.core.manager import ManagedSaveAttempt, ManagedSaveGame, ManagerStore
from rl_fzerox.core.manager.errors import ManagerNameConflictError
from rl_fzerox.core.manager.models import SaveGameStatus


def save_games_payload(store: ManagerStore) -> dict[str, list[dict[str, object]]]:
    items = store.list_save_games()
    return {"save_games": [_save_game_payload(store, item) for item in items]}


def create_save_game_payload(
    store: ManagerStore,
    request: CreateSaveGameRequest,
    name: str,
) -> dict[str, dict[str, object]]:
    try:
        del request
        save_game = store.create_save_game(name=name)
    except ManagerNameConflictError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {"save_game": _save_game_payload(store, save_game)}


def update_save_game_payload(
    store: ManagerStore,
    save_game_id: str,
    request: UpdateSaveGameRequest,
) -> dict[str, dict[str, object]]:
    try:
        save_game = store.rename_save_game(save_game_id=save_game_id, name=request.name)
    except ManagerNameConflictError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    if save_game is None:
        raise HTTPException(status_code=404, detail="save game not found")
    return {"save_game": _save_game_payload(store, save_game)}


def upsert_save_course_setup_payload(
    store: ManagerStore,
    save_game_id: str,
    request: UpsertSaveCourseSetupRequest,
) -> dict[str, dict[str, object]]:
    try:
        store.upsert_save_course_setup(
            save_game_id=save_game_id,
            scope=request.scope,
            difficulty=request.difficulty,
            cup_id=request.cup_id,
            course_id=request.course_id,
            policy_run_id=request.policy_run_id,
            policy_artifact=request.policy_artifact,
            vehicle_id=request.vehicle_id,
            engine_setting_raw_value=request.engine_setting_raw_value,
        )
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error).strip("'")) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    save_game = store.get_save_game(save_game_id)
    if save_game is None:
        raise HTTPException(status_code=404, detail="save game not found")
    return {"save_game": _save_game_payload(store, save_game)}


def start_next_save_attempt_payload(
    store: ManagerStore,
    save_game_id: str,
) -> dict[str, dict[str, object]]:
    try:
        store.start_next_save_attempt(save_game_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error).strip("'")) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    save_game = store.get_save_game(save_game_id)
    if save_game is None:
        raise HTTPException(status_code=404, detail="save game not found")
    return {
        "save_game": _save_game_payload(
            store,
            save_game,
            cleanup_orphan_runner=False,
        )
    }


def save_attempt_execution_context_payload_for_attempt(
    store: ManagerStore,
    attempt_id: str,
) -> dict[str, object]:
    try:
        context = store.get_save_attempt_execution_context(attempt_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error).strip("'")) from error
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    if context is None:
        raise HTTPException(status_code=404, detail="save attempt not found")
    return {"execution_context": save_attempt_execution_context_payload(context)}


def save_attempt_execution_plan_payload_for_attempt(
    store: ManagerStore,
    attempt_id: str,
) -> dict[str, object]:
    try:
        context = store.get_save_attempt_execution_context(attempt_id)
        if context is None:
            raise HTTPException(status_code=404, detail="save attempt not found")
        plan = build_save_race_execution_plan(context)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error).strip("'")) from error
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {"execution_plan": save_race_execution_plan_report(context=context, plan=plan)}


def open_save_game_dir_payload(store: ManagerStore, save_game_id: str) -> dict[str, bool]:
    save_game = store.get_save_game(save_game_id)
    if save_game is None:
        raise HTTPException(status_code=404, detail="save game not found")
    try:
        open_directory(save_game.save_path.parent)
    except RuntimeError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {"opened": True}


def start_career_mode_payload(
    launcher: RunLauncher,
    save_game_id: str,
    request: StartCareerModeRequest,
) -> dict[str, str]:
    try:
        status = launcher.start_career_mode(
            save_game_id=save_game_id,
            device=request.device,
            renderer=request.renderer,
            attempt_seed=request.attempt_seed,
            deterministic_policy=request.policy_mode == "deterministic",
            target_kind=request.target_kind,
            difficulty=request.difficulty,
            cup_id=request.cup_id,
            course_id=request.course_id,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {"status": status}


def _save_game_payload(
    store: ManagerStore,
    save_game: ManagedSaveGame,
    *,
    cleanup_orphan_runner: bool = True,
) -> dict[str, object]:
    runner_active = _career_runner_active(store, save_game)
    unlock_progress = store.save_game_unlock_progress(save_game.id)
    attempts = store.list_save_attempts(save_game.id)
    if (
        cleanup_orphan_runner
        and any(attempt.status == "running" for attempt in attempts)
        and not runner_active
    ):
        store.discard_running_save_attempts(save_game_id=save_game.id)
        updated = store.update_save_game_status(
            save_game_id=save_game.id,
            status=_status_after_orphan_runner_cleanup(
                attempts=attempts,
                completed_targets=unlock_progress.completed_count,
            ),
        )
        if updated is not None:
            save_game = updated
        unlock_progress = store.save_game_unlock_progress(save_game.id)
        attempts = store.list_save_attempts(save_game.id)
    if _should_reset_stale_unstarted_status(
        save_game=save_game,
        runner_active=runner_active,
        attempts=attempts,
        completed_targets=unlock_progress.completed_count,
    ):
        updated = store.update_save_game_status(save_game_id=save_game.id, status="created")
        if updated is not None:
            save_game = updated
    return save_game_payload(
        save_game,
        runner_active=runner_active,
        unlock_progress=unlock_progress,
        attempts=attempts,
        course_setups=store.list_save_course_setups(save_game.id),
    )


def _status_after_orphan_runner_cleanup(
    *,
    attempts: tuple[ManagedSaveAttempt, ...],
    completed_targets: int,
) -> SaveGameStatus:
    has_real_attempt = any(_real_terminal_race_attempt(attempt) for attempt in attempts)
    if completed_targets == 0 and not has_real_attempt:
        return "created"
    return "paused"


def _should_reset_stale_unstarted_status(
    *,
    save_game: ManagedSaveGame,
    runner_active: bool,
    attempts: tuple[ManagedSaveAttempt, ...],
    completed_targets: int,
) -> bool:
    if runner_active or save_game.status not in {"running", "paused"}:
        return False
    return completed_targets == 0 and not any(
        _real_terminal_race_attempt(attempt) for attempt in attempts
    )


def _real_terminal_race_attempt(attempt: ManagedSaveAttempt) -> bool:
    if attempt.status == "running" or attempt.finished_at is None:
        return False
    if attempt.status == "succeeded":
        return True
    return attempt.failure_reason in {"crashed", "depleted", "retired"}


def _career_runner_active(store: ManagerStore, save_game: ManagedSaveGame) -> bool:
    lease_id = store.viewer_lease_id(kind="career_mode", owner_id=save_game.id)
    return (
        active_career_mode_runner_pid(
            store=store,
            lease_id=lease_id,
            save_game_id=save_game.id,
        )
        is not None
    )
