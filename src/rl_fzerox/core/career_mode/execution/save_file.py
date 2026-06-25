# src/rl_fzerox/core/career_mode/execution/save_file.py
"""Save-RAM load and persist helpers for Career Mode runs.

This module stays below watch/runtime config binding. Callers pass an explicit
save path or store/save-game id so managed SQLite state remains owned by the
manager/runtime layer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from rl_fzerox.core.manager.models import ManagedSaveGame


class SaveRamSession(Protocol):
    def read_save_ram(self) -> bytes: ...

    def write_save_ram(self, data: bytes) -> None: ...


class SaveRamRuntimeSession(Protocol):
    @property
    def emulator(self) -> SaveRamSession: ...


class SaveRamGameStore(Protocol):
    def get_save_game(self, save_game_id: str) -> ManagedSaveGame | None: ...


def load_save_ram_from_path(save_path: Path, session: SaveRamRuntimeSession) -> None:
    if save_path.is_file():
        session.emulator.write_save_ram(save_path.read_bytes())


def persist_save_ram_for_store(
    store: SaveRamGameStore,
    save_game_id: str,
    session: SaveRamRuntimeSession,
) -> None:
    persist_save_ram_to_path(save_path_for_store(store, save_game_id), session)


def persist_save_ram_to_path(save_path: Path, session: SaveRamRuntimeSession) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_bytes(session.emulator.read_save_ram())


def save_path_for_store(store: SaveRamGameStore, save_game_id: str) -> Path:
    save_game = store.get_save_game(save_game_id)
    if save_game is None:
        raise RuntimeError("career mode save game disappeared")
    return save_game.save_path
