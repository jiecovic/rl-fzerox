# src/rl_fzerox/core/manager/db/repositories/save_games/__init__.py
"""Facade for save-game repository operations."""

from __future__ import annotations

from rl_fzerox.core.manager.db.repositories.save_games.attempts import (
    delete_running_save_attempts,
    fail_running_save_attempts,
    finish_save_attempt,
    get_save_attempt,
    insert_save_attempt,
    list_save_attempts,
    running_save_attempt,
)
from rl_fzerox.core.manager.db.repositories.save_games.mapping import (
    course_setup_from_model,
    cup_setup_from_model,
    save_attempt_from_model,
    save_game_from_model,
)
from rl_fzerox.core.manager.db.repositories.save_games.records import (
    assert_save_game_name_available,
    delete_save_game,
    get_save_game,
    insert_save_game,
    list_save_games,
    rename_save_game,
    touch_save_game,
    update_save_game_runner_settings,
    update_save_game_status,
)
from rl_fzerox.core.manager.db.repositories.save_games.setups import (
    list_course_setups,
    list_cup_setups,
    upsert_course_setup,
    upsert_cup_setup,
)

__all__ = [
    "assert_save_game_name_available",
    "course_setup_from_model",
    "cup_setup_from_model",
    "delete_running_save_attempts",
    "delete_save_game",
    "fail_running_save_attempts",
    "finish_save_attempt",
    "get_save_attempt",
    "get_save_game",
    "insert_save_attempt",
    "insert_save_game",
    "list_course_setups",
    "list_cup_setups",
    "list_save_attempts",
    "list_save_games",
    "rename_save_game",
    "running_save_attempt",
    "save_attempt_from_model",
    "save_game_from_model",
    "touch_save_game",
    "update_save_game_runner_settings",
    "update_save_game_status",
    "upsert_course_setup",
    "upsert_cup_setup",
]
