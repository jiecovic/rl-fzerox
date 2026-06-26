# src/rl_fzerox/core/manager/db/repositories/save_games/mapping.py
"""ORM-to-domain mapping for save-game repository rows."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from rl_fzerox.core.domain.engine import engine_percent_to_slider_step
from rl_fzerox.core.manager.db.models.save_games import (
    SaveGameAttemptModel,
    SaveGameCourseSetupModel,
    SaveGameCupSetupModel,
    SaveGameModel,
)
from rl_fzerox.core.manager.models import (
    ManagedSaveAttempt,
    ManagedSaveCourseSetup,
    ManagedSaveCupSetup,
    ManagedSaveGame,
    PolicySourceArtifact,
    PolicySourceKind,
)
from rl_fzerox.core.manager.registry.common import (
    optional_float,
    optional_int,
    save_attempt_status,
    save_game_status,
)


def save_game_from_model(row: SaveGameModel) -> ManagedSaveGame:
    """Convert one ORM row into a domain save-game record."""

    return ManagedSaveGame(
        id=row.id,
        name=row.name,
        status=save_game_status(row.status),
        save_path=Path(row.save_path).expanduser().resolve(),
        created_at=row.created_at,
        updated_at=row.updated_at,
        last_finished_at=row.last_finished_at,
        runner_device=_required_runner_device(row.runner_device),
        runner_renderer=_required_runner_renderer(row.runner_renderer),
        runner_policy_mode=_required_policy_mode(row.runner_policy_mode),
        runner_attempt_seed=optional_int(row.runner_attempt_seed),
        runner_recording_enabled=bool(row.runner_recording_enabled),
        runner_recording_input_hud_enabled=bool(row.runner_recording_input_hud_enabled),
        runner_recording_upscale_factor=_required_recording_upscale_factor(
            row.runner_recording_upscale_factor
        ),
        runner_recording_path=None
        if row.runner_recording_path is None
        else Path(row.runner_recording_path).expanduser(),
        runner_target_restart_on_retire=bool(row.runner_target_restart_on_retire),
        runner_target_clear_goal=_required_target_clear_goal(row.runner_target_clear_goal),
        runner_keep_failed_recordings=bool(row.runner_keep_failed_recordings),
        runner_reload_policy_between_attempts=bool(row.runner_reload_policy_between_attempts),
    )


def course_setup_from_model(
    row: SaveGameCourseSetupModel,
) -> ManagedSaveCourseSetup:
    """Convert one ORM row into a domain course setup."""

    engine_setting_raw_value = optional_int(row.engine_setting_raw_value)
    return ManagedSaveCourseSetup(
        id=row.id,
        save_game_id=row.save_game_id,
        policy_source_kind=_required_policy_source_kind(row.policy_source_kind),
        policy_source_id=row.policy_source_id,
        policy_artifact=_required_policy_artifact(row.policy_artifact),
        engine_setting_raw_value=engine_percent_to_slider_step(50)
        if engine_setting_raw_value is None
        else engine_setting_raw_value,
        created_at=row.created_at,
        updated_at=row.updated_at,
        difficulty=row.difficulty,
        cup_id=row.cup_id,
        course_id=row.course_id,
    )


def cup_setup_from_model(
    row: SaveGameCupSetupModel,
) -> ManagedSaveCupSetup:
    """Convert one ORM row into a domain cup setup."""

    return ManagedSaveCupSetup(
        id=row.id,
        save_game_id=row.save_game_id,
        cup_id=row.cup_id,
        vehicle_id=row.vehicle_id,
        created_at=row.created_at,
        updated_at=row.updated_at,
        difficulty=row.difficulty,
    )


def save_attempt_from_model(row: SaveGameAttemptModel) -> ManagedSaveAttempt:
    """Convert one ORM row into a domain attempt record."""

    return ManagedSaveAttempt(
        id=row.id,
        save_game_id=row.save_game_id,
        status=save_attempt_status(row.status),
        target_kind=row.target_kind,
        difficulty=row.difficulty,
        cup_id=row.cup_id,
        course_id=row.course_id,
        started_at=row.started_at,
        finished_at=row.finished_at,
        finish_position=optional_int(row.finish_position),
        finish_time_s=optional_float(row.finish_time_s),
        failure_reason=row.failure_reason,
    )


def _required_policy_source_kind(value: object) -> PolicySourceKind:
    match str(value):
        case "run":
            return "run"
        case "evaluation":
            return "evaluation"
        case "checkpoint":
            return "checkpoint"
    raise ValueError(f"course setup has invalid policy source kind: {value!r}")


def _required_policy_artifact(value: object) -> PolicySourceArtifact:
    match str(value):
        case "latest":
            return "latest"
        case "best":
            return "best"
    raise ValueError(f"course setup has invalid policy artifact: {value!r}")


def _required_runner_device(value: object) -> Literal["cpu", "cuda"]:
    if value == "cpu":
        return "cpu"
    if value == "cuda":
        return "cuda"
    raise ValueError(f"save game has invalid runner device: {value!r}")


def _required_runner_renderer(value: object) -> Literal["angrylion", "gliden64"]:
    if value == "angrylion":
        return "angrylion"
    if value == "gliden64":
        return "gliden64"
    raise ValueError(f"save game has invalid runner renderer: {value!r}")


def _required_policy_mode(value: object) -> Literal["deterministic", "stochastic"]:
    if value == "deterministic":
        return "deterministic"
    if value == "stochastic":
        return "stochastic"
    raise ValueError(f"save game has invalid runner policy mode: {value!r}")


def _required_recording_upscale_factor(value: object) -> int:
    factor = optional_int(value)
    if factor is not None and 1 <= factor <= 4:
        return factor
    raise ValueError(f"save game has invalid recording upscale factor: {value!r}")


def _required_target_clear_goal(value: object) -> int:
    parsed = optional_int(value)
    if parsed is None:
        return 1
    return max(0, parsed)
