# src/rl_fzerox/core/manager/store.py
"""Public store facade over manager registry, storage bootstrap, and artifacts."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from rl_fzerox.core.career_mode.runner.context import SaveAttemptExecutionContext
from rl_fzerox.core.manager.artifacts.paths import (
    manager_runs_root,
    manager_save_games_root,
    manager_tensorboard_views_root,
)
from rl_fzerox.core.manager.artifacts.tensorboard_views import (
    TensorboardViewGroup,
    rebuild_tensorboard_views,
)
from rl_fzerox.core.manager.db import manager_engine
from rl_fzerox.core.manager.models import (
    CourseSetupScope,
    ManagedRun,
    ManagedRunDraft,
    ManagedRunEvent,
    ManagedRunSummary,
    ManagedRunTemplate,
    ManagedSaveAttempt,
    ManagedSaveCourseSetup,
    ManagedSaveGame,
    ManagedSaveUnlockProgress,
    ManagedViewerLease,
    RunCommand,
    RunStatus,
    SaveAttemptStatus,
    SaveGameStatus,
    ViewerLeaseKind,
)
from rl_fzerox.core.manager.registry import drafts as draft_registry
from rl_fzerox.core.manager.registry import lineages as lineage_registry
from rl_fzerox.core.manager.registry import runs as run_registry
from rl_fzerox.core.manager.registry import save_games as save_game_registry
from rl_fzerox.core.manager.registry import viewers as viewer_registry
from rl_fzerox.core.manager.registry.common import new_run_id
from rl_fzerox.core.manager.run_spec import ManagedRunConfig
from rl_fzerox.core.manager.storage.schema import (
    initialize_manager_schema,
    refresh_default_template,
)

if TYPE_CHECKING:
    from rl_fzerox.core.training.session.callbacks.track_sampling import (
        TrackSamplingRuntimeState,
    )


def default_manager_db_path() -> Path:
    """Return the default local manager registry path."""

    return Path("local/manager/runs.db").resolve()


def new_managed_run_id(name: str) -> str:
    """Return one stable opaque id for a managed run."""

    del name
    return new_run_id()


class ManagerStore:
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
        with self._connect() as connection:
            lineage_registry.backfill_lineage_ids(connection)
            lineage_registry.migrate_lineage_layout_rows(
                connection,
                migrated_at=self.utc_now(),
            )
        self._schema_initialized = True

    def refresh_system_templates(self) -> None:
        """Refresh built-in templates without re-running full schema bootstrap."""

        self._ensure_schema_initialized()
        refresh_default_template(self.db_path, updated_at=self.utc_now())

    def manager_runs_root(self, *, output_root: Path | None = None) -> Path:
        return manager_runs_root(output_root=output_root)

    def tensorboard_views_root(self, *, output_root: Path | None = None) -> Path:
        if output_root is None:
            return self.db_path.parent.parent / "tensorboard_views"
        return manager_tensorboard_views_root(output_root=output_root)

    def save_games_root(self, *, output_root: Path | None = None) -> Path:
        if output_root is None:
            return self.db_path.parent.parent / "save_games"
        return manager_save_games_root(output_root=output_root)

    def path(self, value: str | Path) -> Path:
        return Path(value).expanduser().resolve()

    def create_run(
        self,
        *,
        run_id: str | None = None,
        name: str,
        config: ManagedRunConfig,
        managed_runs_root: Path | None = None,
        explicit_run_dir: Path | None = None,
        lineage_id: str | None = None,
        lineage_step_offset: int = 0,
        parent_run_id: str | None = None,
        source_run_id: str | None = None,
        source_artifact: Literal["latest", "best"] | None = None,
        source_snapshot_dir: Path | None = None,
        source_num_timesteps: int | None = None,
        exclude_draft_id: str | None = None,
    ) -> ManagedRun:
        return run_registry.create_run(
            self,
            run_id=run_id,
            name=name,
            config=config,
            managed_runs_root=managed_runs_root,
            explicit_run_dir=explicit_run_dir,
            lineage_id=lineage_id,
            lineage_step_offset=lineage_step_offset,
            parent_run_id=parent_run_id,
            source_run_id=source_run_id,
            source_artifact=source_artifact,
            source_snapshot_dir=source_snapshot_dir,
            source_num_timesteps=source_num_timesteps,
            exclude_draft_id=exclude_draft_id,
        )

    def get_run(self, run_id: str) -> ManagedRun | None:
        return run_registry.get_run(self, run_id)

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

    def list_runs(self) -> tuple[ManagedRun, ...]:
        return run_registry.list_runs(self)

    def list_visible_runs(self) -> tuple[ManagedRun, ...]:
        return run_registry.list_visible_runs(self)

    def list_visible_run_summaries(self) -> tuple[ManagedRunSummary, ...]:
        return run_registry.list_visible_run_summaries(self)

    def list_recent_run_events(
        self,
        run_ids: tuple[str, ...],
        *,
        limit_per_run: int = 6,
    ) -> dict[str, tuple[ManagedRunEvent, ...]]:
        return run_registry.list_recent_run_events(
            self,
            run_ids,
            limit_per_run=limit_per_run,
        )

    def update_run_status(
        self,
        *,
        run_id: str,
        status: RunStatus,
        message: str,
        started_at: str | None = None,
        stopped_at: str | None = None,
    ) -> ManagedRun | None:
        return run_registry.update_run_status(
            self,
            run_id=run_id,
            status=status,
            message=message,
            started_at=started_at,
            stopped_at=stopped_at,
        )

    def clear_run_runtime(self, run_id: str) -> None:
        run_registry.clear_run_runtime(self, run_id)

    def clear_run_track_sampling_state(self, run_id: str) -> None:
        run_registry.clear_run_track_sampling_state(self, run_id)

    def get_run_track_sampling_state(
        self,
        run_id: str,
    ) -> TrackSamplingRuntimeState | None:
        return run_registry.get_run_track_sampling_state(self, run_id)

    def upsert_run_track_sampling_state(
        self,
        *,
        run_id: str,
        state: TrackSamplingRuntimeState,
        updated_at: str | None = None,
    ) -> None:
        run_registry.upsert_run_track_sampling_state(
            self,
            run_id=run_id,
            state=state,
            updated_at=updated_at,
        )

    def register_run_worker(
        self,
        *,
        run_id: str,
        launch_token: str,
        pid: int,
        launched_at: str,
    ) -> bool:
        return run_registry.register_run_worker(
            self,
            run_id=run_id,
            launch_token=launch_token,
            pid=pid,
            launched_at=launched_at,
        )

    def heartbeat_run_worker(
        self,
        *,
        run_id: str,
        launch_token: str,
        heartbeat_at: str,
    ) -> bool:
        return run_registry.heartbeat_run_worker(
            self,
            run_id=run_id,
            launch_token=launch_token,
            heartbeat_at=heartbeat_at,
        )

    def clear_run_worker(self, run_id: str, *, launch_token: str | None = None) -> None:
        run_registry.clear_run_worker(self, run_id, launch_token=launch_token)

    def upsert_run_runtime(
        self,
        *,
        run_id: str,
        total_timesteps: int,
        num_timesteps: int,
        progress_fraction: float,
        updated_at: str,
        fps: float | None = None,
        episode_reward_mean: float | None = None,
        episode_length_mean: float | None = None,
        approx_kl: float | None = None,
        entropy_loss: float | None = None,
        value_loss: float | None = None,
        policy_gradient_loss: float | None = None,
    ) -> None:
        run_registry.upsert_run_runtime(
            self,
            run_id=run_id,
            total_timesteps=total_timesteps,
            num_timesteps=num_timesteps,
            progress_fraction=progress_fraction,
            updated_at=updated_at,
            fps=fps,
            episode_reward_mean=episode_reward_mean,
            episode_length_mean=episode_length_mean,
            approx_kl=approx_kl,
            entropy_loss=entropy_loss,
            value_loss=value_loss,
            policy_gradient_loss=policy_gradient_loss,
        )

    def append_run_event(
        self,
        *,
        run_id: str,
        kind: str,
        message: str,
        created_at: str | None = None,
    ) -> None:
        run_registry.append_run_event(
            self,
            run_id=run_id,
            kind=kind,
            message=message,
            created_at=created_at,
        )

    def update_run_fork_source(
        self,
        *,
        run_id: str,
        source_snapshot_dir: Path,
        source_num_timesteps: int,
        lineage_step_offset: int | None = None,
    ) -> ManagedRun | None:
        return run_registry.update_run_fork_source(
            self,
            run_id=run_id,
            source_snapshot_dir=source_snapshot_dir,
            source_num_timesteps=source_num_timesteps,
            lineage_step_offset=lineage_step_offset,
        )

    def request_run_command(
        self,
        *,
        run_id: str,
        command: RunCommand,
    ) -> ManagedRun | None:
        return run_registry.request_run_command(self, run_id=run_id, command=command)

    def pending_run_command(self, run_id: str) -> RunCommand | None:
        return run_registry.pending_run_command(self, run_id)

    def clear_run_command(
        self,
        run_id: str,
        *,
        command: RunCommand | None = None,
    ) -> None:
        run_registry.clear_run_command(self, run_id, command=command)

    def list_drafts(self) -> tuple[ManagedRunDraft, ...]:
        return draft_registry.list_drafts(self)

    def delete_draft(self, draft_id: str) -> bool:
        return draft_registry.delete_draft(self, draft_id)

    def delete_run(self, run_id: str) -> bool:
        return lineage_registry.delete_run(self, run_id)

    def delete_lineage(self, lineage_id: str) -> bool:
        return lineage_registry.delete_lineage(self, lineage_id)

    def update_lineage_groups(
        self,
        *,
        lineage_id: str,
        group_names: tuple[str, ...],
    ) -> tuple[str, ...]:
        return lineage_registry.update_lineage_groups(
            self,
            lineage_id=lineage_id,
            group_names=group_names,
        )

    def rebuild_tensorboard_views(self) -> tuple[TensorboardViewGroup, ...]:
        return rebuild_tensorboard_views(
            self.list_visible_runs(),
            view_root=self.tensorboard_views_root(),
        )

    def migrate_lineage_layout(self) -> int:
        return lineage_registry.migrate_lineage_layout(self)

    def update_run_name(self, *, run_id: str, name: str) -> ManagedRun | None:
        return run_registry.update_run_name(self, run_id=run_id, name=name)

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

    def create_save_game(
        self,
        *,
        name: str,
        save_games_root: Path | None = None,
    ) -> ManagedSaveGame:
        return save_game_registry.create_save_game(
            self,
            name=name,
            save_games_root=save_games_root or self.save_games_root(),
        )

    def get_save_game(self, save_game_id: str) -> ManagedSaveGame | None:
        return save_game_registry.get_save_game(self, save_game_id)

    def list_save_games(self) -> tuple[ManagedSaveGame, ...]:
        return save_game_registry.list_save_games(self)

    def rename_save_game(
        self,
        *,
        save_game_id: str,
        name: str,
    ) -> ManagedSaveGame | None:
        return save_game_registry.rename_save_game(
            self,
            save_game_id=save_game_id,
            name=name,
        )

    def save_game_unlock_progress(self, save_game_id: str) -> ManagedSaveUnlockProgress:
        return save_game_registry.unlock_progress(self, save_game_id)

    def list_save_course_setups(
        self,
        save_game_id: str,
    ) -> tuple[ManagedSaveCourseSetup, ...]:
        return save_game_registry.list_course_setups(self, save_game_id)

    def start_save_attempt(
        self,
        *,
        save_game_id: str,
        target_kind: str | None = None,
        policy_run_id: str | None = None,
        policy_artifact: Literal["latest", "best"] | None = None,
        difficulty: str | None = None,
        cup_id: str | None = None,
        course_id: str | None = None,
    ) -> ManagedSaveAttempt:
        return save_game_registry.start_save_attempt(
            self,
            save_game_id=save_game_id,
            target_kind=target_kind,
            policy_run_id=policy_run_id,
            policy_artifact=policy_artifact,
            difficulty=difficulty,
            cup_id=cup_id,
            course_id=course_id,
        )

    def start_next_save_attempt(
        self,
        save_game_id: str,
    ) -> ManagedSaveAttempt:
        return save_game_registry.start_next_save_attempt(self, save_game_id)

    def start_or_reuse_next_save_attempt(
        self,
        save_game_id: str,
    ) -> ManagedSaveAttempt:
        return save_game_registry.start_or_reuse_next_save_attempt(self, save_game_id)

    def list_save_attempts(
        self,
        save_game_id: str,
    ) -> tuple[ManagedSaveAttempt, ...]:
        return save_game_registry.list_save_attempts(self, save_game_id)

    def fail_running_save_attempts(
        self,
        *,
        save_game_id: str,
        failure_reason: str,
    ) -> int:
        return save_game_registry.fail_running_save_attempts(
            self,
            save_game_id=save_game_id,
            failure_reason=failure_reason,
        )

    def discard_running_save_attempts(
        self,
        *,
        save_game_id: str,
    ) -> int:
        return save_game_registry.discard_running_save_attempts(
            self,
            save_game_id=save_game_id,
        )

    def get_save_attempt_execution_context(
        self,
        attempt_id: str,
    ) -> SaveAttemptExecutionContext | None:
        return save_game_registry.get_save_attempt_execution_context(self, attempt_id)

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
            self,
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
        scope: CourseSetupScope,
        policy_run_id: str,
        policy_artifact: Literal["latest", "best"],
        difficulty: str | None = None,
        cup_id: str | None = None,
        course_id: str | None = None,
    ) -> ManagedSaveCourseSetup:
        return save_game_registry.upsert_course_setup(
            self,
            save_game_id=save_game_id,
            scope=scope,
            policy_run_id=policy_run_id,
            policy_artifact=policy_artifact,
            difficulty=difficulty,
            cup_id=cup_id,
            course_id=course_id,
        )

    def update_save_game_status(
        self,
        *,
        save_game_id: str,
        status: SaveGameStatus,
    ) -> ManagedSaveGame | None:
        return save_game_registry.update_save_game_status(
            self,
            save_game_id=save_game_id,
            status=status,
        )

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

    def clear_viewer_lease(
        self,
        *,
        lease_id: str,
        pid: int | None = None,
    ) -> bool:
        return viewer_registry.clear_viewer_lease(self, lease_id=lease_id, pid=pid)

    def reconcile_orphaned_runs(self) -> None:
        run_registry.reconcile_orphaned_runs(self)

    def _drain_pending_filesystem_operations(self) -> None:
        run_registry.drain_pending_filesystem_operations(self)

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
    def _orm_session(self) -> Iterator[Session]:
        with Session(self._manager_engine(), expire_on_commit=False) as session:
            with session.begin():
                yield session
