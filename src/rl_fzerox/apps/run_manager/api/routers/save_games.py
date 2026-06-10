# src/rl_fzerox/apps/run_manager/api/routers/save_games.py
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, HTTPException, Path

from rl_fzerox.apps.run_manager.api import handlers
from rl_fzerox.apps.run_manager.api.contracts import (
    CreateSaveGameRequest,
    RunLauncher,
    StartCareerModeRequest,
    UpdateSaveGameRequest,
    UpsertSaveCourseSetupRequest,
)
from rl_fzerox.apps.run_manager.api.execution import run_sync
from rl_fzerox.core.manager import ManagerStore


def create_save_games_router(store: ManagerStore, launcher: RunLauncher) -> APIRouter:
    router = APIRouter()

    @router.get("/api/save-games")
    async def save_games() -> dict[str, list[dict[str, object]]]:
        return await run_sync(handlers.save_games_payload, store)

    @router.post("/api/save-games", status_code=201)
    async def create_save_game(
        request: CreateSaveGameRequest,
    ) -> dict[str, dict[str, object]]:
        name = request.name.strip()
        if not name:
            raise HTTPException(status_code=400, detail="save-game name is required")
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
        name = request.name.strip()
        if not name:
            raise HTTPException(status_code=400, detail="save-game name is required")
        return await run_sync(
            handlers.update_save_game_payload,
            store,
            save_game_id,
            UpdateSaveGameRequest(name=name),
        )

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
