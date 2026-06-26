# src/rl_fzerox/apps/run_manager/api/handlers/save_games.py
from __future__ import annotations

from pathlib import Path

from fastapi import HTTPException

from rl_fzerox.apps.run_manager.api.contracts import (
    CreateSaveGameRequest,
    ImportSaveEngineTuningRequest,
    UpdateSaveGameRequest,
    UpdateSaveRunnerSettingsRequest,
    UpsertSaveCourseSetupRequest,
    UpsertSaveCupSetupRequest,
)
from rl_fzerox.apps.run_manager.api.handlers.save_game_status import (
    save_game_payload_for_store,
    save_game_status_payload_for_store,
)
from rl_fzerox.apps.run_manager.desktop import open_directory
from rl_fzerox.core.engine_tuning import (
    EngineTuningContext,
    OrderedEngineTuner,
)
from rl_fzerox.core.engine_tuning.config import engine_tuner_settings
from rl_fzerox.core.manager import (
    ManagedPolicySource,
    ManagedRun,
    ManagerStore,
    PolicySourceArtifact,
)
from rl_fzerox.core.manager.errors import ManagerNameConflictError
from rl_fzerox.core.manager.projection.engine_tuning import adaptive_engine_tuning_config
from rl_fzerox.core.training.runs import resolve_policy_artifact_path
from rl_fzerox.core.training.session.artifacts import load_engine_tuning_checkpoint_state


def save_games_payload(store: ManagerStore) -> dict[str, list[dict[str, object]]]:
    items = store.list_save_games()
    return {"save_games": [save_game_payload_for_store(store, item) for item in items]}


def save_game_status_payload_for_id(
    store: ManagerStore,
    save_game_id: str,
) -> dict[str, dict[str, object]]:
    save_game = store.get_save_game(save_game_id)
    if save_game is None:
        raise HTTPException(status_code=404, detail="save game not found")
    return {"save_game": save_game_status_payload_for_store(store, save_game)}


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


def update_save_game_runner_settings_payload(
    store: ManagerStore,
    save_game_id: str,
    request: UpdateSaveRunnerSettingsRequest,
) -> dict[str, dict[str, object]]:
    try:
        save_game = store.update_save_game_runner_settings(
            save_game_id=save_game_id,
            device=request.device,
            renderer=request.renderer,
            policy_mode=request.policy_mode,
            attempt_seed=request.attempt_seed,
            recording_enabled=request.recording_enabled,
            recording_input_hud_enabled=request.recording_input_hud_enabled,
            recording_upscale_factor=request.recording_upscale_factor,
            recording_path=request.recording_path,
            target_restart_on_retire=request.target_restart_on_retire,
            target_clear_goal=request.target_clear_goal,
            keep_failed_recordings=request.keep_failed_recordings,
            reload_policy_between_attempts=request.reload_policy_between_attempts,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    if save_game is None:
        raise HTTPException(status_code=404, detail="save game not found")
    return {"save_game": save_game_payload_for_store(store, save_game)}


def delete_save_game_payload(store: ManagerStore, save_game_id: str) -> dict[str, bool]:
    try:
        deleted = store.delete_save_game(save_game_id)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    if not deleted:
        raise HTTPException(status_code=404, detail="save game not found")
    return {"deleted": True}


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
            policy_source_kind=request.policy_source_kind,
            policy_source_id=request.policy_source_id,
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
    try:
        policy_source = store.resolve_policy_source(
            policy_source_kind=request.policy_source_kind,
            policy_source_id=request.policy_source_id,
            policy_artifact=request.policy_artifact,
            require_policy_artifact=request.policy_source_kind == "evaluation",
        )
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error).strip("'")) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    if policy_source.config.vehicle.engine_mode != "adaptive_tuner":
        raise HTTPException(status_code=400, detail="policy source has no adaptive engine tuning")
    try:
        policy_path = _resolve_engine_tuning_policy_path(store, policy_source)
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=400,
            detail=f"policy source has no {request.policy_artifact} artifact",
        ) from error
    state = load_engine_tuning_checkpoint_state(policy_path)
    if state is None or (not state.candidates and state.model_state is None):
        raise HTTPException(status_code=400, detail="policy source has no engine tuning samples")
    if not request.course_setups:
        raise HTTPException(
            status_code=400,
            detail="no course setup drafts use this policy source",
        )
    tuner = OrderedEngineTuner(
        settings=engine_tuner_settings(adaptive_engine_tuning_config(policy_source.config)),
        state=state,
    )
    recommendations: list[dict[str, object]] = []
    for setup in request.course_setups:
        recommendation = tuner.recommendation(
            EngineTuningContext(
                course_key=setup.course_id,
                vehicle_id=setup.vehicle_id,
            )
        )
        recommendations.append(
            {
                "difficulty": setup.difficulty,
                "cup_id": setup.cup_id,
                "course_id": setup.course_id,
                "vehicle_id": setup.vehicle_id,
                "engine_setting_raw_value": recommendation.engine_setting_raw_value,
                "mean_score": recommendation.mean_score,
                "finish_count": recommendation.finish_count,
            }
        )
    return {"recommendations": recommendations}


def open_save_game_dir_payload(store: ManagerStore, save_game_id: str) -> dict[str, bool]:
    save_game = store.get_save_game(save_game_id)
    if save_game is None:
        raise HTTPException(status_code=404, detail="save game not found")
    try:
        open_directory(save_game.save_path.parent)
    except RuntimeError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {"opened": True}


def _resolve_engine_tuning_policy_path(
    store: ManagerStore,
    policy_source: ManagedPolicySource,
) -> Path:
    if policy_source.kind == "evaluation":
        if policy_source.policy_path is None:
            raise FileNotFoundError("evaluation policy checkpoint is missing")
        return policy_source.policy_path
    run = store.get_run(policy_source.id)
    if run is None:
        raise FileNotFoundError(f"policy run not found: {policy_source.id}")
    return _resolve_engine_tuning_run_policy_artifact(run, artifact=policy_source.artifact)


def _resolve_engine_tuning_run_policy_artifact(
    run: ManagedRun,
    *,
    artifact: PolicySourceArtifact,
) -> Path:
    try:
        return resolve_policy_artifact_path(run.run_dir, artifact=artifact)
    except FileNotFoundError:
        if run.source_snapshot_dir is None:
            raise
    return resolve_policy_artifact_path(run.source_snapshot_dir, artifact=artifact)
