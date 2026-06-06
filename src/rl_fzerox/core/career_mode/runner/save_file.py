# src/rl_fzerox/core/career_mode/runner/save_file.py
from __future__ import annotations

from pathlib import Path
from typing import Protocol

from rl_fzerox.core.manager import ManagerStore
from rl_fzerox.core.manager.models import ManagedSaveGame
from rl_fzerox.core.runtime_spec.schema import WatchAppConfig


class SaveRamSession(Protocol):
    def read_save_ram(self) -> bytes: ...

    def write_save_ram(self, data: bytes) -> None: ...


class SaveRamRuntimeSession(Protocol):
    @property
    def emulator(self) -> SaveRamSession: ...


class SaveRamGameStore(Protocol):
    def get_save_game(self, save_game_id: str) -> ManagedSaveGame | None: ...


def load_save_ram(config: WatchAppConfig, session: SaveRamRuntimeSession) -> None:
    save_path = save_path_from_config(config)
    if save_path.is_file():
        session.emulator.write_save_ram(save_path.read_bytes())


def persist_save_ram(config: WatchAppConfig, session: SaveRamRuntimeSession) -> None:
    persist_save_ram_to_path(save_path_from_config(config), session)


def persist_save_ram_for_store(
    store: SaveRamGameStore,
    save_game_id: str,
    session: SaveRamRuntimeSession,
) -> None:
    save_game = store.get_save_game(save_game_id)
    if save_game is None:
        raise RuntimeError("career mode save game disappeared")
    persist_save_ram_to_path(save_game.save_path, session)


def persist_save_ram_to_path(save_path: Path, session: SaveRamRuntimeSession) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_bytes(session.emulator.read_save_ram())


def save_path_from_config(config: WatchAppConfig) -> Path:
    store = store_from_config(config)
    save_game = store.get_save_game(save_game_id_from_config(config))
    if save_game is None:
        raise RuntimeError("career mode save game disappeared")
    return save_game.save_path


def store_from_config(config: WatchAppConfig) -> ManagerStore:
    db_path = config.watch.manager_db_path
    if db_path is None:
        raise RuntimeError("career mode requires watch.manager_db_path")
    return ManagerStore(db_path)


def save_game_id_from_config(config: WatchAppConfig) -> str:
    save_game_id = config.watch.managed_save_game_id
    if save_game_id is None:
        raise RuntimeError("career mode requires watch.managed_save_game_id")
    return save_game_id
