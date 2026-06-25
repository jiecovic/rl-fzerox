# src/rl_fzerox/ui/watch/runtime/career_mode/save_ram.py
"""Watch-runtime binding for managed Career Mode save RAM.

Core save-RAM helpers operate on explicit paths. This module resolves the
managed run config to the SQLite-backed save game once, then passes only the
resolved path/store identity through the watch loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rl_fzerox.core.career_mode.execution.save_file import (
    SaveRamRuntimeSession,
    load_save_ram_from_path,
    persist_save_ram_to_path,
)
from rl_fzerox.core.manager import ManagerStore
from rl_fzerox.core.runtime_spec.schema import WatchAppConfig


@dataclass(frozen=True, slots=True)
class CareerModeSaveBinding:
    store: ManagerStore
    save_game_id: str
    save_path: Path


def career_mode_save_binding_from_config(config: WatchAppConfig) -> CareerModeSaveBinding:
    db_path = config.watch.manager_db_path
    if db_path is None:
        raise RuntimeError("career mode requires watch.manager_db_path")
    save_game_id = config.watch.managed_save_game_id
    if save_game_id is None:
        raise RuntimeError("career mode requires watch.managed_save_game_id")

    store = ManagerStore(db_path)
    save_game = store.get_save_game(save_game_id)
    if save_game is None:
        raise RuntimeError("career mode save game disappeared")
    return CareerModeSaveBinding(
        store=store,
        save_game_id=save_game_id,
        save_path=save_game.save_path,
    )


def load_career_mode_save_ram(
    binding: CareerModeSaveBinding,
    session: SaveRamRuntimeSession,
) -> None:
    load_save_ram_from_path(binding.save_path, session)


def persist_career_mode_save_ram(
    binding: CareerModeSaveBinding,
    session: SaveRamRuntimeSession,
) -> None:
    persist_save_ram_to_path(binding.save_path, session)
