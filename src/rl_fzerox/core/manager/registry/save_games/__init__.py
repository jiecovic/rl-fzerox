# src/rl_fzerox/core/manager/registry/save_games/__init__.py
"""Facade for manager-store save-game operations."""

from __future__ import annotations

from rl_fzerox.core.manager.registry.save_games.attempts import (
    discard_running_save_attempts,
    fail_running_save_attempts,
    finish_save_attempt,
    get_save_attempt_execution_context,
    list_save_attempts,
    start_next_save_attempt,
    start_or_reuse_next_save_attempt,
    start_save_attempt,
    start_target_save_attempt,
)
from rl_fzerox.core.manager.registry.save_games.records import (
    create_save_game,
    delete_save_game,
    get_save_game,
    list_save_games,
    rename_save_game,
    unlock_progress,
    update_runner_settings,
    update_save_game_status,
)
from rl_fzerox.core.manager.registry.save_games.setups import (
    list_course_setups,
    list_cup_setups,
    upsert_course_setup,
    upsert_cup_setup,
)

__all__ = [
    "create_save_game",
    "delete_save_game",
    "discard_running_save_attempts",
    "fail_running_save_attempts",
    "finish_save_attempt",
    "get_save_attempt_execution_context",
    "get_save_game",
    "list_course_setups",
    "list_cup_setups",
    "list_save_attempts",
    "list_save_games",
    "rename_save_game",
    "start_next_save_attempt",
    "start_or_reuse_next_save_attempt",
    "start_save_attempt",
    "start_target_save_attempt",
    "unlock_progress",
    "update_runner_settings",
    "update_save_game_status",
    "upsert_course_setup",
    "upsert_cup_setup",
]
