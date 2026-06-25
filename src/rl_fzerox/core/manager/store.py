# src/rl_fzerox/core/manager/store.py
"""Public store facade over manager registry, storage bootstrap, and artifacts."""

from __future__ import annotations

import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from rl_fzerox.core.manager.artifacts.tensorboard_views import TensorboardViewGroup
from rl_fzerox.core.manager.db import manager_engine
from rl_fzerox.core.manager.models import (
    ManagedRunDraft,
    ManagedRunTemplate,
    ManagedViewerLease,
    ViewerLeaseKind,
)
from rl_fzerox.core.manager.registry import drafts as draft_registry
from rl_fzerox.core.manager.registry import paths as path_registry
from rl_fzerox.core.manager.registry import tensorboard as tensorboard_registry
from rl_fzerox.core.manager.registry import viewers as viewer_registry
from rl_fzerox.core.manager.registry.common import new_run_id
from rl_fzerox.core.manager.run_spec import ManagedRunConfig
from rl_fzerox.core.manager.storage.schema import (
    initialize_manager_schema,
    refresh_default_template,
)
from rl_fzerox.core.manager.store_api import (
    EvaluationStoreMixin,
    RunStoreMixin,
    SaveGameStoreMixin,
)


def default_manager_db_path() -> Path:
    """Return the default local manager registry path."""

    return Path("local/manager/runs.db").resolve()


def new_managed_run_id(name: str) -> str:
    """Return one stable opaque id for a managed run."""

    del name
    return new_run_id()


class ManagerStore(RunStoreMixin, EvaluationStoreMixin, SaveGameStoreMixin):
    """SQLite-backed source of truth for managed training runs.

    The store owns database lifecycle and exposes a narrow domain API, while
    concrete SQL and filesystem behavior lives in registry/storage/artifact
    subpackages.
    """

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = (db_path or default_manager_db_path()).expanduser().resolve()
        self._schema_initialized = False
        self._orm_engine: Engine | None = None

    def close(self) -> None:
        """Release database resources owned by this store instance."""

        if self._orm_engine is None:
            return
        self._orm_engine.dispose()
        self._orm_engine = None

    def initialize(self) -> None:
        """Create the manager database schema if needed."""

        self._ensure_schema_initialized()
        self._drain_pending_filesystem_operations()

    def _ensure_schema_initialized(self) -> None:
        """Initialize schema once for hot-path store operations."""

        if self._schema_initialized:
            return
        self._initialize_schema()

    def _initialize_schema(self) -> None:
        """Apply schema/bootstrap work to the manager database."""

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        initialize_manager_schema(self.db_path, applied_at=self.utc_now())
        self._schema_initialized = True

    def refresh_system_templates(self) -> None:
        """Refresh built-in templates without re-running full schema bootstrap."""

        self._ensure_schema_initialized()
        refresh_default_template(self.db_path, updated_at=self.utc_now())

    def manager_runs_root(self, *, output_root: Path | None = None) -> Path:
        return path_registry.manager_root(output_root=output_root)

    def tensorboard_views_root(self, *, output_root: Path | None = None) -> Path:
        return path_registry.tensorboard_views_root(self.db_path, output_root=output_root)

    def save_games_root(self, *, output_root: Path | None = None) -> Path:
        return path_registry.save_games_root(self.db_path, output_root=output_root)

    def evaluations_root(self, *, output_root: Path | None = None) -> Path:
        return path_registry.evaluations_root(self.db_path, output_root=output_root)

    def path(self, value: str | Path) -> Path:
        return path_registry.resolved_path(value)

    def create_draft(
        self,
        *,
        name: str,
        config: ManagedRunConfig,
        source_run_id: str | None = None,
        source_artifact: Literal["latest", "best"] | None = None,
    ) -> ManagedRunDraft:
        return draft_registry.create_draft(
            self,
            name=name,
            config=config,
            source_run_id=source_run_id,
            source_artifact=source_artifact,
        )

    def list_drafts(self) -> tuple[ManagedRunDraft, ...]:
        return draft_registry.list_drafts(self)

    def delete_draft(self, draft_id: str) -> bool:
        return draft_registry.delete_draft(self, draft_id)

    def rebuild_tensorboard_views(self) -> tuple[TensorboardViewGroup, ...]:
        return tensorboard_registry.rebuild_tensorboard_views(self)

    def update_draft(
        self,
        *,
        draft_id: str,
        name: str,
        config: ManagedRunConfig,
        source_run_id: str | None = None,
        source_artifact: Literal["latest", "best"] | None = None,
    ) -> ManagedRunDraft | None:
        return draft_registry.update_draft(
            self,
            draft_id=draft_id,
            name=name,
            config=config,
            source_run_id=source_run_id,
            source_artifact=source_artifact,
        )

    def get_draft(self, draft_id: str) -> ManagedRunDraft | None:
        return draft_registry.get_draft(self, draft_id)

    def list_templates(self) -> tuple[ManagedRunTemplate, ...]:
        return draft_registry.list_templates(self)

    def default_template(self) -> ManagedRunTemplate:
        return draft_registry.default_template(self)

    def viewer_lease_id(
        self,
        *,
        kind: ViewerLeaseKind,
        owner_id: str,
        qualifier: str | None = None,
    ) -> str:
        return viewer_registry.viewer_lease_id(
            kind=kind,
            owner_id=owner_id,
            qualifier=qualifier,
        )

    def upsert_viewer_lease(
        self,
        *,
        lease_id: str,
        kind: ViewerLeaseKind,
        owner_id: str,
        pid: int,
        qualifier: str | None = None,
    ) -> ManagedViewerLease:
        return viewer_registry.upsert_viewer_lease(
            self,
            lease_id=lease_id,
            kind=kind,
            owner_id=owner_id,
            pid=pid,
            qualifier=qualifier,
        )

    def get_viewer_lease(self, lease_id: str) -> ManagedViewerLease | None:
        return viewer_registry.get_viewer_lease(self, lease_id)

    def heartbeat_viewer_lease(
        self,
        *,
        lease_id: str,
        pid: int,
        heartbeat_at: str,
    ) -> bool:
        return viewer_registry.heartbeat_viewer_lease(
            self,
            lease_id=lease_id,
            pid=pid,
            heartbeat_at=heartbeat_at,
        )

    def clear_viewer_lease(
        self,
        *,
        lease_id: str,
        pid: int | None = None,
    ) -> bool:
        return viewer_registry.clear_viewer_lease(self, lease_id=lease_id, pid=pid)

    @staticmethod
    def utc_now() -> str:
        from rl_fzerox.core.manager.registry.common import utc_now

        return utc_now()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA busy_timeout = 30000")
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _manager_engine(self) -> Engine:
        if self._orm_engine is None:
            self._orm_engine = manager_engine(self.db_path)
        return self._orm_engine

    @contextmanager
    def _orm_session(self) -> Generator[Session]:
        with Session(self._manager_engine(), expire_on_commit=False) as session:
            with session.begin():
                yield session
