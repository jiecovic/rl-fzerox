# src/rl_fzerox/core/manager/store_api/save_games.py
"""Save-game workflow methods mixed into the public ManagerStore facade."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from rl_fzerox.core.career_mode.execution.context import SaveAttemptExecutionContext
from rl_fzerox.core.domain.engine import engine_percent_to_slider_step
from rl_fzerox.core.manager.models import (
    ManagedSaveAttempt,
    ManagedSaveCourseSetup,
    ManagedSaveCupSetup,
    ManagedSaveGame,
    ManagedSaveUnlockProgress,
    SaveAttemptStatus,
    SaveGameStatus,
)
from rl_fzerox.core.manager.registry import save_games as save_game_registry
from rl_fzerox.core.manager.store_api.common import manager_store as _manager_store


class SaveGameStoreMixin:
    """ManagerStore facade methods for managed save-game workflows."""

    def create_save_game(
        self,
        *,
        name: str,
        save_games_root: Path | None = None,
    ) -> ManagedSaveGame:
        return save_game_registry.create_save_game(
            _manager_store(self),
            name=name,
            save_games_root=save_games_root or _manager_store(self).save_games_root(),
        )

    def get_save_game(self, save_game_id: str) -> ManagedSaveGame | None:
        return save_game_registry.get_save_game(_manager_store(self), save_game_id)

    def list_save_games(self) -> tuple[ManagedSaveGame, ...]:
        return save_game_registry.list_save_games(_manager_store(self))

    def rename_save_game(
        self,
        *,
        save_game_id: str,
        name: str,
    ) -> ManagedSaveGame | None:
        return save_game_registry.rename_save_game(
            _manager_store(self),
            save_game_id=save_game_id,
            name=name,
        )

    def update_save_game_runner_settings(
        self,
        *,
        save_game_id: str,
        device: Literal["cpu", "cuda"],
        renderer: Literal["angrylion", "gliden64"],
        policy_mode: Literal["deterministic", "stochastic"],
        attempt_seed: int | None,
        recording_enabled: bool,
        recording_input_hud_enabled: bool,
        recording_upscale_factor: int,
        recording_path: Path | None,
        target_restart_on_retire: bool,
        target_clear_goal: int,
        keep_failed_recordings: bool,
        reload_policy_between_attempts: bool,
    ) -> ManagedSaveGame | None:
        return save_game_registry.update_runner_settings(
            _manager_store(self),
            save_game_id=save_game_id,
            device=device,
            renderer=renderer,
            policy_mode=policy_mode,
            attempt_seed=attempt_seed,
            recording_enabled=recording_enabled,
            recording_input_hud_enabled=recording_input_hud_enabled,
            recording_upscale_factor=recording_upscale_factor,
            recording_path=recording_path,
            target_restart_on_retire=target_restart_on_retire,
            target_clear_goal=target_clear_goal,
            keep_failed_recordings=keep_failed_recordings,
            reload_policy_between_attempts=reload_policy_between_attempts,
        )

    def delete_save_game(self, save_game_id: str) -> bool:
        return save_game_registry.delete_save_game(_manager_store(self), save_game_id)

    def save_game_unlock_progress(
        self,
        save_game_id: str,
    ) -> ManagedSaveUnlockProgress:
        return save_game_registry.unlock_progress(_manager_store(self), save_game_id)

    def list_save_course_setups(
        self,
        save_game_id: str,
    ) -> tuple[ManagedSaveCourseSetup, ...]:
        return save_game_registry.list_course_setups(_manager_store(self), save_game_id)

    def list_save_cup_setups(
        self,
        save_game_id: str,
    ) -> tuple[ManagedSaveCupSetup, ...]:
        return save_game_registry.list_cup_setups(_manager_store(self), save_game_id)

    def start_save_attempt(
        self,
        *,
        save_game_id: str,
        target_kind: str | None = None,
        difficulty: str | None = None,
        cup_id: str | None = None,
        course_id: str | None = None,
    ) -> ManagedSaveAttempt:
        return save_game_registry.start_save_attempt(
            _manager_store(self),
            save_game_id=save_game_id,
            target_kind=target_kind,
            difficulty=difficulty,
            cup_id=cup_id,
            course_id=course_id,
        )

    def start_next_save_attempt(
        self,
        save_game_id: str,
    ) -> ManagedSaveAttempt:
        return save_game_registry.start_next_save_attempt(_manager_store(self), save_game_id)

    def start_target_save_attempt(
        self,
        save_game_id: str,
        *,
        target_kind: str,
        difficulty: str,
        cup_id: str,
        course_id: str | None = None,
    ) -> ManagedSaveAttempt:
        return save_game_registry.start_target_save_attempt(
            _manager_store(self),
            save_game_id,
            target_kind=target_kind,
            difficulty=difficulty,
            cup_id=cup_id,
            course_id=course_id,
        )

    def start_or_reuse_next_save_attempt(
        self,
        save_game_id: str,
    ) -> ManagedSaveAttempt:
        return save_game_registry.start_or_reuse_next_save_attempt(
            _manager_store(self), save_game_id
        )

    def list_save_attempts(
        self,
        save_game_id: str,
    ) -> tuple[ManagedSaveAttempt, ...]:
        return save_game_registry.list_save_attempts(_manager_store(self), save_game_id)

    def fail_running_save_attempts(
        self,
        *,
        save_game_id: str,
        failure_reason: str,
    ) -> int:
        return save_game_registry.fail_running_save_attempts(
            _manager_store(self),
            save_game_id=save_game_id,
            failure_reason=failure_reason,
        )

    def discard_running_save_attempts(
        self,
        *,
        save_game_id: str,
    ) -> int:
        return save_game_registry.discard_running_save_attempts(
            _manager_store(self),
            save_game_id=save_game_id,
        )

    def get_save_attempt_execution_context(
        self,
        attempt_id: str,
    ) -> SaveAttemptExecutionContext | None:
        return save_game_registry.get_save_attempt_execution_context(
            _manager_store(self), attempt_id
        )

    def finish_save_attempt(
        self,
        *,
        attempt_id: str,
        status: SaveAttemptStatus,
        finish_position: int | None = None,
        finish_time_s: float | None = None,
        failure_reason: str | None = None,
    ) -> ManagedSaveAttempt | None:
        return save_game_registry.finish_save_attempt(
            _manager_store(self),
            attempt_id=attempt_id,
            status=status,
            finish_position=finish_position,
            finish_time_s=finish_time_s,
            failure_reason=failure_reason,
        )

    def upsert_save_course_setup(
        self,
        *,
        save_game_id: str,
        policy_run_id: str,
        policy_artifact: Literal["latest", "best"],
        engine_setting_raw_value: int = engine_percent_to_slider_step(50),
        difficulty: str | None = None,
        cup_id: str | None = None,
        course_id: str | None = None,
    ) -> ManagedSaveCourseSetup:
        return save_game_registry.upsert_course_setup(
            _manager_store(self),
            save_game_id=save_game_id,
            policy_run_id=policy_run_id,
            policy_artifact=policy_artifact,
            engine_setting_raw_value=engine_setting_raw_value,
            difficulty=difficulty,
            cup_id=cup_id,
            course_id=course_id,
        )

    def upsert_save_cup_setup(
        self,
        *,
        save_game_id: str,
        cup_id: str,
        vehicle_id: str,
        difficulty: str | None = None,
    ) -> ManagedSaveCupSetup:
        return save_game_registry.upsert_cup_setup(
            _manager_store(self),
            save_game_id=save_game_id,
            cup_id=cup_id,
            vehicle_id=vehicle_id,
            difficulty=difficulty,
        )

    def update_save_game_status(
        self,
        *,
        save_game_id: str,
        status: SaveGameStatus,
    ) -> ManagedSaveGame | None:
        return save_game_registry.update_save_game_status(
            _manager_store(self),
            save_game_id=save_game_id,
            status=status,
        )
