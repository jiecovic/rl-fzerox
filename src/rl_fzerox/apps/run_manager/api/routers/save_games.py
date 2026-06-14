# src/rl_fzerox/apps/run_manager/api/routers/save_games.py
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Path

from rl_fzerox.apps.run_manager.api import handlers
from rl_fzerox.apps.run_manager.api.contracts import (
    CreateSaveGameRequest,
    ImportSaveEngineTuningRequest,
    RunLauncher,
    StartCareerModeRequest,
    UpdateSaveGameRequest,
    UpsertSaveCourseSetupRequest,
    UpsertSaveCupSetupRequest,
)
from rl_fzerox.apps.run_manager.api.execution import run_sync
from rl_fzerox.apps.run_manager.api.validation import required_name
from rl_fzerox.core.manager import ManagerStore


def create_save_games_router(store: ManagerStore, launcher: RunLauncher) -> APIRouter:
    router = APIRouter()

    @router.get("/api/save-games")
    async def save_games() -> dict[str, list[dict[str, object]]]:
        return await run_sync(handlers.save_games_payload, store)

    @router.get("/api/save-games/{save_game_id}/status")
    async def save_game_status(
        save_game_id: Annotated[str, Path(min_length=1)],
    ) -> dict[str, dict[str, object]]:
        return await run_sync(handlers.save_game_status_payload_for_id, store, save_game_id)

    @router.post("/api/save-games", status_code=201)
    async def create_save_game(
        request: CreateSaveGameRequest,
    ) -> dict[str, dict[str, object]]:
        name = required_name(request.name, subject="save-game")
        return await run_sync(handlers.create_save_game_payload, store, request, name)

    @router.post("/api/save-games/{save_game_id}/open-dir")
    async def open_save_game_dir(
        save_game_id: Annotated[str, Path(min_length=1)],
    ) -> dict[str, bool]:
        return await run_sync(handlers.open_save_game_dir_payload, store, save_game_id)

    @router.put("/api/save-games/{save_game_id}")
    async def update_save_game(
        save_game_id: Annotated[str, Path(min_length=1)],
        request: UpdateSaveGameRequest,
    ) -> dict[str, dict[str, object]]:
        name = required_name(request.name, subject="save-game")
        return await run_sync(
            handlers.update_save_game_payload,
            store,
            save_game_id,
            UpdateSaveGameRequest(name=name),
        )

    @router.delete("/api/save-games/{save_game_id}")
    async def delete_save_game(
        save_game_id: Annotated[str, Path(min_length=1)],
    ) -> dict[str, bool]:
        return await run_sync(handlers.delete_save_game_payload, store, save_game_id)

    @router.post("/api/save-games/{save_game_id}/runner")
    async def start_career_mode(
        save_game_id: Annotated[str, Path(min_length=1)],
        request: StartCareerModeRequest | None = None,
    ) -> dict[str, str]:
        return await run_sync(
            handlers.start_career_mode_payload,
            launcher,
            save_game_id,
            StartCareerModeRequest() if request is None else request,
        )

    @router.put("/api/save-games/{save_game_id}/course-setups")
    async def upsert_save_course_setup(
        save_game_id: Annotated[str, Path(min_length=1)],
        request: UpsertSaveCourseSetupRequest,
    ) -> dict[str, dict[str, object]]:
        return await run_sync(
            handlers.upsert_save_course_setup_payload,
            store,
            save_game_id,
            request,
        )

    @router.put("/api/save-games/{save_game_id}/cup-setups")
    async def upsert_save_cup_setup(
        save_game_id: Annotated[str, Path(min_length=1)],
        request: UpsertSaveCupSetupRequest,
    ) -> dict[str, dict[str, object]]:
        return await run_sync(
            handlers.upsert_save_cup_setup_payload,
            store,
            save_game_id,
            request,
        )

    @router.post("/api/save-games/{save_game_id}/course-setups/import-engine-tuning")
    async def import_save_engine_tuning(
        save_game_id: Annotated[str, Path(min_length=1)],
        request: ImportSaveEngineTuningRequest,
    ) -> dict[str, object]:
        return await run_sync(
            handlers.import_save_engine_tuning_payload,
            store,
            save_game_id,
            request,
        )

    @router.post("/api/save-games/{save_game_id}/attempts/next")
    async def start_next_save_attempt(
        save_game_id: Annotated[str, Path(min_length=1)],
    ) -> dict[str, dict[str, object]]:
        return await run_sync(
            handlers.start_next_save_attempt_payload,
            store,
            save_game_id,
        )

    @router.get("/api/save-attempts/{attempt_id}/execution-context")
    async def save_attempt_execution_context(
        attempt_id: Annotated[str, Path(min_length=1)],
    ) -> dict[str, object]:
        return await run_sync(
            handlers.save_attempt_execution_context_payload_for_attempt,
            store,
            attempt_id,
        )

    @router.get("/api/save-attempts/{attempt_id}/execution-plan")
    async def save_attempt_execution_plan(
        attempt_id: Annotated[str, Path(min_length=1)],
    ) -> dict[str, object]:
        return await run_sync(
            handlers.save_attempt_execution_plan_payload_for_attempt,
            store,
            attempt_id,
        )

    return router
