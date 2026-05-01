# src/rl_fzerox/core/manager/models.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from rl_fzerox.core.manager.config import ManagedRunConfig

RunStatus = Literal["created", "running", "paused", "stopped", "finished", "failed"]


@dataclass(frozen=True, slots=True)
class ManagedRun:
    """One immutable DB-managed run record."""

    id: str
    name: str
    status: RunStatus
    config: ManagedRunConfig
    config_hash: str
    run_dir: Path
    created_at: str
    parent_run_id: str | None = None
    source_run_id: str | None = None
    source_artifact: str | None = None
    started_at: str | None = None
    stopped_at: str | None = None


@dataclass(frozen=True, slots=True)
class ManagedRunTemplate:
    """One DB-backed editable starting point for future immutable runs."""

    id: str
    name: str
    config: ManagedRunConfig
    config_hash: str
    created_at: str
    updated_at: str


@dataclass(frozen=True, slots=True)
class ManagedRunDraft:
    """One SQLite-only editable run configuration draft."""

    id: str
    name: str
    config: ManagedRunConfig
    config_hash: str
    created_at: str
    updated_at: str
