# src/rl_fzerox/apps/run_manager/api/payloads/save_games.py
from __future__ import annotations

from rl_fzerox.core.career_mode.execution.context import SaveAttemptExecutionContext
from rl_fzerox.core.manager import (
    ManagedSaveAttempt,
    ManagedSaveCourseSetup,
    ManagedSaveCupSetup,
    ManagedSaveGame,
    ManagedSaveUnlockProgress,
    ManagedSaveUnlockTarget,
)


def save_course_setup_payload(
    assignment: ManagedSaveCourseSetup,
) -> dict[str, object]:
    return {
        "id": assignment.id,
        "save_game_id": assignment.save_game_id,
        "difficulty": assignment.difficulty,
        "cup_id": assignment.cup_id,
        "course_id": assignment.course_id,
        "policy_source_kind": assignment.policy_source_kind,
        "policy_source_id": assignment.policy_source_id,
        "policy_artifact": assignment.policy_artifact,
        "engine_setting_raw_value": assignment.engine_setting_raw_value,
        "created_at": assignment.created_at,
        "updated_at": assignment.updated_at,
    }


def save_cup_setup_payload(
    assignment: ManagedSaveCupSetup,
) -> dict[str, object]:
    return {
        "id": assignment.id,
        "save_game_id": assignment.save_game_id,
        "difficulty": assignment.difficulty,
        "cup_id": assignment.cup_id,
        "vehicle_id": assignment.vehicle_id,
        "created_at": assignment.created_at,
        "updated_at": assignment.updated_at,
    }


def save_attempt_payload(attempt: ManagedSaveAttempt) -> dict[str, object]:
    return {
        "id": attempt.id,
        "save_game_id": attempt.save_game_id,
        "target_kind": attempt.target_kind,
        "status": attempt.status,
        "difficulty": attempt.difficulty,
        "cup_id": attempt.cup_id,
        "course_id": attempt.course_id,
        "started_at": attempt.started_at,
        "finished_at": attempt.finished_at,
        "finish_position": attempt.finish_position,
        "finish_time_s": attempt.finish_time_s,
        "failure_reason": attempt.failure_reason,
    }


def save_game_runner_settings_payload(save_game: ManagedSaveGame) -> dict[str, object]:
    return {
        "device": save_game.runner_device,
        "renderer": save_game.runner_renderer,
        "policy_mode": save_game.runner_policy_mode,
        "attempt_seed": save_game.runner_attempt_seed,
        "recording_enabled": save_game.runner_recording_enabled,
        "recording_input_hud_enabled": save_game.runner_recording_input_hud_enabled,
        "recording_upscale_factor": save_game.runner_recording_upscale_factor,
        "recording_path": None
        if save_game.runner_recording_path is None
        else str(save_game.runner_recording_path),
        "target_restart_on_retire": save_game.runner_target_restart_on_retire,
        "target_clear_goal": save_game.runner_target_clear_goal,
        "keep_failed_recordings": save_game.runner_keep_failed_recordings,
        "reload_policy_between_attempts": save_game.runner_reload_policy_between_attempts,
    }


def save_attempt_execution_context_payload(
    context: SaveAttemptExecutionContext,
) -> dict[str, object]:
    """Serialize one resolved career-runner attempt context."""

    return {
        "save_game": save_game_payload(context.save_game),
        "attempt": save_attempt_payload(context.attempt),
        "target": {
            "kind": context.target.kind,
            "label": context.target.label,
            "difficulty": context.target.difficulty,
            "cup_id": context.target.cup_id,
            "course_id": context.target.course_id,
        },
        "course_setup": save_course_setup_payload(context.course_setup),
        "cup_setup": save_cup_setup_payload(context.cup_setup),
        "policy_source": {
            "kind": context.policy_source.kind,
            "id": context.policy_source.id,
            "name": context.policy_source.name,
            "mutable": context.policy_source.mutable,
            "source_run_id": context.policy_source.source_run_id,
            "source_run_name": context.policy_source.source_run_name,
        },
        "policy_artifact": context.policy_artifact,
        "policy_path": str(context.policy_path),
    }


def save_unlock_target_payload(target: ManagedSaveUnlockTarget) -> dict[str, object]:
    return {
        "sequence_index": target.sequence_index,
        "kind": target.kind,
        "status": target.status,
        "label": target.label,
        "difficulty": target.difficulty,
        "cup_id": target.cup_id,
        "course_id": target.course_id,
    }


def save_unlock_progress_payload(progress: ManagedSaveUnlockProgress) -> dict[str, object]:
    return {
        "inspection_status": progress.inspection_status,
        "completed_count": progress.completed_count,
        "total_count": progress.total_count,
        "unlocked_vehicle_count": progress.unlocked_vehicle_count,
        "unlocked_vehicle_ids": list(progress.unlocked_vehicle_ids),
        "next_target": None
        if progress.next_target is None
        else save_unlock_target_payload(progress.next_target),
        "targets": [save_unlock_target_payload(target) for target in progress.targets],
    }


def save_game_payload(
    save_game: ManagedSaveGame,
    *,
    runner_active: bool = False,
    unlock_progress: ManagedSaveUnlockProgress | None = None,
    attempts: tuple[ManagedSaveAttempt, ...] = (),
    course_setups: tuple[ManagedSaveCourseSetup, ...] = (),
    cup_setups: tuple[ManagedSaveCupSetup, ...] = (),
) -> dict[str, object]:
    return {
        "id": save_game.id,
        "name": save_game.name,
        "status": save_game.status,
        "save_path": str(save_game.save_path),
        "created_at": save_game.created_at,
        "updated_at": save_game.updated_at,
        "last_finished_at": save_game.last_finished_at,
        "runner_active": runner_active,
        "runner_settings": save_game_runner_settings_payload(save_game),
        "unlock_progress": None
        if unlock_progress is None
        else save_unlock_progress_payload(unlock_progress),
        "attempts": [save_attempt_payload(attempt) for attempt in attempts],
        "course_setups": [save_course_setup_payload(assignment) for assignment in course_setups],
        "cup_setups": [save_cup_setup_payload(assignment) for assignment in cup_setups],
    }


def save_game_status_payload(
    save_game: ManagedSaveGame,
    *,
    runner_active: bool = False,
    unlock_progress: ManagedSaveUnlockProgress | None = None,
) -> dict[str, object]:
    return {
        "id": save_game.id,
        "name": save_game.name,
        "status": save_game.status,
        "save_path": str(save_game.save_path),
        "created_at": save_game.created_at,
        "updated_at": save_game.updated_at,
        "last_finished_at": save_game.last_finished_at,
        "runner_active": runner_active,
        "runner_settings": save_game_runner_settings_payload(save_game),
        "unlock_progress": None
        if unlock_progress is None
        else save_unlock_progress_payload(unlock_progress),
    }
