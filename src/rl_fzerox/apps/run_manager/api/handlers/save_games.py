# src/rl_fzerox/apps/run_manager/api/handlers/save_games.py
from __future__ import annotations

from fastapi import HTTPException

from rl_fzerox.apps.run_manager.api.contracts import (
    CreateSaveGameRequest,
    UpdateSaveGameRequest,
    UpsertSaveCourseSetupRequest,
)
from rl_fzerox.apps.run_manager.api.handlers.save_game_status import (
    save_game_payload_for_store,
)
from rl_fzerox.apps.run_manager.desktop import open_directory
from rl_fzerox.core.manager import ManagerStore
from rl_fzerox.core.manager.errors import ManagerNameConflictError


def save_games_payload(store: ManagerStore) -> dict[str, list[dict[str, object]]]:
    items = store.list_save_games()
    return {"save_games": [save_game_payload_for_store(store, item) for item in items]}


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
    return {"save_game": save_game_payload_for_store(store, save_game)}


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
    return {"save_game": save_game_payload_for_store(store, save_game)}


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
    return {"save_game": save_game_payload_for_store(store, save_game)}


def open_save_game_dir_payload(store: ManagerStore, save_game_id: str) -> dict[str, bool]:
    save_game = store.get_save_game(save_game_id)
    if save_game is None:
        raise HTTPException(status_code=404, detail="save game not found")
    try:
        open_directory(save_game.save_path.parent)
    except RuntimeError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {"opened": True}
