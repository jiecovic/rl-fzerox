# src/rl_fzerox/core/manager/registry/save_games/records.py
"""Save-game record, settings, status, and progress operations."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

from rl_fzerox.core.career_mode.progress.unlocks import build_unlock_progress
from rl_fzerox.core.manager.artifacts.paths import predicted_managed_save_game_path
from rl_fzerox.core.manager.db.repositories import save_games as save_game_repository
from rl_fzerox.core.manager.db.repositories.filesystem import queue_delete_tree
from rl_fzerox.core.manager.models import (
    ManagedSaveGame,
    ManagedSaveUnlockProgress,
    SaveGameStatus,
)
from rl_fzerox.core.manager.registry.common import new_record_id, utc_now

if TYPE_CHECKING:
    from rl_fzerox.core.manager.store import ManagerStore


def create_save_game(
    store: ManagerStore,
    *,
    name: str,
    save_games_root: Path | None = None,
) -> ManagedSaveGame:
    """Create one manager-owned save-game record."""

    if not name.strip():
        raise ValueError("save-game name is required")
    store.initialize()
    created_at = utc_now()
    save_game_id = new_record_id(name)
    normalized_name = name.strip()
    save_path = predicted_managed_save_game_path(save_game_id, output_root=save_games_root)
    save_game = ManagedSaveGame(
        id=save_game_id,
        name=normalized_name,
        status="created",
        save_path=save_path,
        created_at=created_at,
        updated_at=created_at,
    )
    with store._orm_session() as session:
        save_game_repository.assert_save_game_name_available(session, normalized_name)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_game_repository.insert_save_game(session, save_game)
    return save_game


def get_save_game(store: ManagerStore, save_game_id: str) -> ManagedSaveGame | None:
    """Return one save game by id."""

    store.initialize()
    with store._orm_session() as session:
        return save_game_repository.get_save_game(session, save_game_id)


def list_save_games(store: ManagerStore) -> tuple[ManagedSaveGame, ...]:
    """Return all save games in manager display order."""

    store.initialize()
    with store._orm_session() as session:
        return save_game_repository.list_save_games(session)


def rename_save_game(
    store: ManagerStore,
    *,
    save_game_id: str,
    name: str,
) -> ManagedSaveGame | None:
    """Rename one manager-owned save game."""

    normalized_name = name.strip()
    if not normalized_name:
        raise ValueError("save-game name is required")
    store.initialize()
    with store._orm_session() as session:
        return save_game_repository.rename_save_game(
            session,
            save_game_id=save_game_id,
            name=normalized_name,
            updated_at=utc_now(),
        )


def update_runner_settings(
    store: ManagerStore,
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
    """Persist Career runner launch settings for one save game."""

    if attempt_seed is not None and not 0 <= attempt_seed <= (1 << 32) - 1:
        raise ValueError("attempt seed must be an integer from 0 to 4294967295")
    if not 1 <= recording_upscale_factor <= 4:
        raise ValueError("recording upscale factor must be an integer from 1 to 4")
    if target_clear_goal < 0:
        raise ValueError("target clear goal must be non-negative")
    store.initialize()
    with store._orm_session() as session:
        return save_game_repository.update_save_game_runner_settings(
            session,
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
            updated_at=utc_now(),
        )


def delete_save_game(store: ManagerStore, save_game_id: str) -> bool:
    """Delete one manager-owned save game and queue its filesystem cleanup."""

    store.initialize()
    deleted_at = utc_now()
    with store._orm_session() as session:
        save_game = save_game_repository.get_save_game(session, save_game_id)
        if save_game is None:
            return False
        if save_game.status == "running":
            raise ValueError("stop or pause the career runner before deleting this save")
        save_root = save_game.save_path.parent
        if save_root.exists():
            queue_delete_tree(session, path=save_root, created_at=deleted_at)
        save_game_repository.delete_save_game(session, save_game_id)
    store._drain_pending_filesystem_operations()
    return True


def unlock_progress(
    store: ManagerStore,
    save_game_id: str,
) -> ManagedSaveUnlockProgress:
    """Return save-file unlock progress for one save game."""

    store.initialize()
    with store._orm_session() as session:
        save_game = save_game_repository.get_save_game(session, save_game_id)
        if save_game is None:
            raise KeyError("save game not found")
        attempts = save_game_repository.list_save_attempts(session, save_game_id)
    return build_unlock_progress(save_game.save_path, attempts=attempts)


def update_save_game_status(
    store: ManagerStore,
    *,
    save_game_id: str,
    status: SaveGameStatus,
) -> ManagedSaveGame | None:
    """Update one save-game lifecycle status."""

    store.initialize()
    now = utc_now()
    last_finished_at = now if status in {"finished", "failed"} else None
    with store._orm_session() as session:
        return save_game_repository.update_save_game_status(
            session,
            save_game_id=save_game_id,
            status=status,
            updated_at=now,
            last_finished_at=last_finished_at,
        )
