# src/rl_fzerox/apps/run_manager/api/handlers/save_game_attempts.py
from __future__ import annotations

from fastapi import HTTPException

from rl_fzerox.apps.run_manager.api.handlers.save_game_status import (
    save_game_payload_for_store,
)
from rl_fzerox.apps.run_manager.api.payloads import save_attempt_execution_context_payload
from rl_fzerox.core.career_mode.runner.race import build_save_race_execution_plan
from rl_fzerox.core.career_mode.runner.reports import save_race_execution_plan_report
from rl_fzerox.core.manager import ManagerStore


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
        "save_game": save_game_payload_for_store(
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
