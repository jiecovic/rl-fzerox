# src/rl_fzerox/apps/run_manager/api/handlers/save_games.py
from __future__ import annotations

from fastapi import HTTPException

from rl_fzerox.apps.run_manager.api.contracts import (
    CreateSaveGameRequest,
    ImportSaveEngineTuningRequest,
    UpdateSaveGameRequest,
    UpsertSaveCourseSetupRequest,
    UpsertSaveCupSetupRequest,
)
from rl_fzerox.apps.run_manager.api.handlers.save_game_status import (
    save_game_payload_for_store,
)
from rl_fzerox.apps.run_manager.desktop import open_directory
from rl_fzerox.core.engine_tuning import (
    AdaptiveEngineBandit,
    EngineTuningContext,
    engine_bandit_settings,
)
from rl_fzerox.core.manager import ManagedRun, ManagerStore
from rl_fzerox.core.manager.errors import ManagerNameConflictError
from rl_fzerox.core.manager.models import ManagedSaveCourseSetup, ManagedSaveCupSetup
from rl_fzerox.core.runtime_spec.schema import AdaptiveEngineTuningConfig
from rl_fzerox.core.training.runs import resolve_policy_artifact_path
from rl_fzerox.core.training.session.artifacts import load_engine_tuning_checkpoint_state


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
            difficulty=request.difficulty,
            cup_id=request.cup_id,
            course_id=request.course_id,
            policy_run_id=request.policy_run_id,
            policy_artifact=request.policy_artifact,
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


def upsert_save_cup_setup_payload(
    store: ManagerStore,
    save_game_id: str,
    request: UpsertSaveCupSetupRequest,
) -> dict[str, dict[str, object]]:
    try:
        store.upsert_save_cup_setup(
            save_game_id=save_game_id,
            difficulty=request.difficulty,
            cup_id=request.cup_id,
            vehicle_id=request.vehicle_id,
        )
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error).strip("'")) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    save_game = store.get_save_game(save_game_id)
    if save_game is None:
        raise HTTPException(status_code=404, detail="save game not found")
    return {"save_game": save_game_payload_for_store(store, save_game)}


def import_save_engine_tuning_payload(
    store: ManagerStore,
    save_game_id: str,
    request: ImportSaveEngineTuningRequest,
) -> dict[str, object]:
    save_game = store.get_save_game(save_game_id)
    if save_game is None:
        raise HTTPException(status_code=404, detail="save game not found")
    run = store.get_run(request.policy_run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="policy run not found")
    if run.config.vehicle.engine_mode != "adaptive_bandit":
        raise HTTPException(status_code=400, detail="policy run has no adaptive engine tuning")
    try:
        policy_path = resolve_policy_artifact_path(
            run.run_dir,
            artifact=request.policy_artifact,
        )
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=400,
            detail=f"policy run has no {request.policy_artifact} artifact",
        ) from error
    state = load_engine_tuning_checkpoint_state(policy_path)
    if state is None or not state.arms:
        raise HTTPException(status_code=400, detail="policy run has no engine tuning samples")

    course_setups = tuple(
        setup
        for setup in store.list_save_course_setups(save_game_id)
        if setup.policy_run_id == run.id
    )
    if not course_setups:
        raise HTTPException(
            status_code=400,
            detail="save game has no course setups using this policy run",
        )
    cup_setups = store.list_save_cup_setups(save_game_id)
    bandit = AdaptiveEngineBandit(
        settings=engine_bandit_settings(_adaptive_engine_config(run)),
        state=state,
    )
    applied = []
    for setup in course_setups:
        vehicle_id = _vehicle_for_course_setup(run, setup, cup_setups)
        recommendation = bandit.recommendation(
            EngineTuningContext(
                course_key=setup.course_id or "unknown",
                vehicle_id=vehicle_id,
            )
        )
        store.upsert_save_course_setup(
            save_game_id=save_game_id,
            difficulty=setup.difficulty,
            cup_id=setup.cup_id,
            course_id=setup.course_id,
            policy_run_id=setup.policy_run_id,
            policy_artifact=request.policy_artifact,
            engine_setting_raw_value=recommendation.engine_setting_raw_value,
        )
        applied.append(
            {
                "setup_id": setup.id,
                "difficulty": setup.difficulty,
                "cup_id": setup.cup_id,
                "course_id": setup.course_id,
                "vehicle_id": vehicle_id,
                "engine_setting_raw_value": recommendation.engine_setting_raw_value,
                "mean_score": recommendation.mean_score,
                "attempts": recommendation.attempts,
            }
        )
    updated = store.get_save_game(save_game_id)
    if updated is None:
        raise HTTPException(status_code=404, detail="save game not found")
    return {
        "applied": applied,
        "save_game": save_game_payload_for_store(store, updated),
    }


def open_save_game_dir_payload(store: ManagerStore, save_game_id: str) -> dict[str, bool]:
    save_game = store.get_save_game(save_game_id)
    if save_game is None:
        raise HTTPException(status_code=404, detail="save game not found")
    try:
        open_directory(save_game.save_path.parent)
    except RuntimeError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {"opened": True}


def _vehicle_for_course_setup(
    run: ManagedRun,
    setup: ManagedSaveCourseSetup,
    cup_setups: tuple[ManagedSaveCupSetup, ...],
) -> str:
    for cup_setup in cup_setups:
        if cup_setup.cup_id != setup.cup_id:
            continue
        if cup_setup.difficulty != setup.difficulty:
            continue
        return cup_setup.vehicle_id
    selected = getattr(run.config.vehicle, "selected_vehicle_ids", ())
    return selected[0] if selected else "blue_falcon"


def _adaptive_engine_config(run: ManagedRun) -> AdaptiveEngineTuningConfig:
    vehicle = run.config.vehicle
    return AdaptiveEngineTuningConfig(
        enabled=True,
        min_raw_value=vehicle.engine_setting_min_raw_value,
        max_raw_value=vehicle.engine_setting_max_raw_value,
        bin_size=vehicle.adaptive_engine_bin_size,
        stat_decay=vehicle.adaptive_engine_stat_decay,
        prior_mean=vehicle.adaptive_engine_prior_mean,
        prior_strength=vehicle.adaptive_engine_prior_strength,
        exploration_scale=vehicle.adaptive_engine_exploration_scale,
        uniform_exploration=vehicle.adaptive_engine_uniform_exploration,
        completion_weight=vehicle.adaptive_engine_completion_weight,
        finish_bonus=vehicle.adaptive_engine_finish_bonus,
        position_weight=vehicle.adaptive_engine_position_weight,
    )
