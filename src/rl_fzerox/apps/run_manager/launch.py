# src/rl_fzerox/apps/run_manager/launch.py
from __future__ import annotations

from pathlib import Path
from typing import Literal

import rl_fzerox.apps.run_manager.launching as launching
from rl_fzerox.core.manager import ManagedRun, ManagedRunConfig, ManagerStore, new_managed_run_id
from rl_fzerox.core.manager.artifacts.fork_source import (
    clone_fork_source,
    is_complete_fork_source,
    run_fork_source_dir,
    snapshot_fork_source,
)
from rl_fzerox.core.manager.artifacts.paths import predicted_managed_run_dir
from rl_fzerox.core.manager.models import RunCommand
from rl_fzerox.core.manager.training import (
    assert_managed_fork_compatible,
    build_managed_fork_train_app_config,
    build_managed_train_app_config,
)
from rl_fzerox.core.training.runs import (
    resolve_model_artifact_path,
)


class ManagerRunLauncher:
    """Launch and control manager-owned training runs."""

    def __init__(self, store: ManagerStore) -> None:
        self._store = store

    def launch(
        self,
        *,
        name: str,
        config: ManagedRunConfig,
        draft_id: str | None = None,
        source_run_id: str | None = None,
        source_artifact: Literal["latest", "best"] | None = None,
    ) -> ManagedRun:
        normalized_name = name.strip()
        if not normalized_name:
            raise ValueError("run name is required")
        if source_run_id is not None and source_artifact is not None:
            if draft_id is None:
                return self.fork(
                    run_id=source_run_id,
                    artifact=source_artifact,
                    name=normalized_name,
                    config=config,
                )
            draft = self._store.get_draft(draft_id)
            if draft is None:
                raise ValueError(f"draft not found: {draft_id}")
            if draft.source_run_id != source_run_id or draft.source_artifact != source_artifact:
                raise ValueError("fork draft source no longer matches the requested source run")
            if draft.source_snapshot_dir is None or draft.source_num_timesteps is None:
                raise ValueError(
                    "fork draft is missing its pinned checkpoint snapshot; recreate the fork draft"
                )
            return self.fork(
                run_id=source_run_id,
                artifact=source_artifact,
                name=normalized_name,
                config=config,
                exclude_draft_id=draft_id,
                source_snapshot_dir=draft.source_snapshot_dir,
                source_num_timesteps=draft.source_num_timesteps,
            )

        run_id = new_managed_run_id(normalized_name)
        run_dir = predicted_managed_run_dir(run_id, lineage_id=run_id)
        train_config = build_managed_train_app_config(
            config,
            run_id=run_id,
            run_dir=run_dir,
        )
        run = self._store.create_run(
            run_id=run_id,
            name=normalized_name,
            config=config,
            explicit_run_dir=run_dir,
            lineage_id=run_id,
            exclude_draft_id=draft_id,
        )
        launching.persist_launch_manifest(run_dir=run.run_dir, train_config=train_config)
        self._spawn_worker(run_id=run.id, resume=False)
        launched = self._store.update_run_status(
            run_id=run.id,
            status="running",
            started_at=launching.utc_now(),
            stopped_at=None,
            message=f"training worker launched; log: {launching.manager_worker_log_path(run.id)}",
        )
        if launched is None:
            raise RuntimeError(f"managed run disappeared during launch: {run.id}")
        return launched

    def fork(
        self,
        *,
        run_id: str,
        artifact: Literal["latest", "best"],
        name: str | None = None,
        config: ManagedRunConfig | None = None,
        exclude_draft_id: str | None = None,
        source_snapshot_dir: Path | None = None,
        source_num_timesteps: int | None = None,
    ) -> ManagedRun:
        """Launch one child run warm-started from a parent run checkpoint."""

        source_run = self._store.get_run(run_id)
        if source_run is None:
            raise ValueError(f"run not found: {run_id}")
        if artifact not in {"latest", "best"}:
            raise ValueError(f"unsupported fork artifact: {artifact}")

        normalized_name = (name or launching.default_fork_name(source_run.name, artifact)).strip()
        if not normalized_name:
            raise ValueError("run name is required")
        child_config = config or source_run.config
        assert_managed_fork_compatible(source_run.config, child_config)
        child_run_id = new_managed_run_id(normalized_name)
        child_run_dir = predicted_managed_run_dir(
            child_run_id,
            lineage_id=source_run.lineage_id,
        )
        child_source_snapshot_dir = run_fork_source_dir(run_dir=child_run_dir)
        if source_snapshot_dir is None or source_num_timesteps is None:
            source_num_timesteps = snapshot_fork_source(
                source_run_dir=source_run.run_dir,
                artifact=artifact,
                destination_dir=child_source_snapshot_dir,
            )
        else:
            clone_fork_source(
                source_dir=source_snapshot_dir,
                destination_dir=child_source_snapshot_dir,
            )
        child_lineage_step_offset = source_run.lineage_step_offset + source_num_timesteps
        train_config = build_managed_fork_train_app_config(
            child_config,
            run_id=child_run_id,
            run_dir=child_run_dir,
            source_run_dir=child_source_snapshot_dir,
            source_artifact=artifact,
            source_config=source_run.config,
            tensorboard_step_offset=child_lineage_step_offset,
        )
        child_run = self._store.create_run(
            run_id=child_run_id,
            name=normalized_name,
            config=child_config,
            explicit_run_dir=child_run_dir,
            lineage_id=source_run.lineage_id,
            lineage_step_offset=child_lineage_step_offset,
            parent_run_id=source_run.id,
            source_run_id=source_run.id,
            source_artifact=artifact,
            source_snapshot_dir=child_source_snapshot_dir,
            source_num_timesteps=source_num_timesteps,
            exclude_draft_id=exclude_draft_id,
        )
        launching.persist_launch_manifest(run_dir=child_run.run_dir, train_config=train_config)
        self._spawn_worker(run_id=child_run.id, resume=False)
        launched = self._store.update_run_status(
            run_id=child_run.id,
            status="running",
            started_at=launching.utc_now(),
            stopped_at=None,
            message=(
                f"forked from {source_run.name} ({artifact} @ {source_num_timesteps:,} steps); "
                f"log: {launching.manager_worker_log_path(child_run.id)}"
            ),
        )
        if launched is None:
            raise RuntimeError(f"managed child run disappeared during launch: {child_run.id}")
        return launched

    def resume(self, *, run_id: str) -> ManagedRun:
        """Resume one paused or stopped run in place from its latest checkpoint."""

        run = self._store.get_run(run_id)
        if run is None:
            raise ValueError(f"run not found: {run_id}")
        if run.status not in {"paused", "stopped", "failed"}:
            raise ValueError("only paused, stopped, or failed runs can be resumed")

        self._store.clear_run_command(run.id)
        reset_local_clock = False
        try:
            resolve_model_artifact_path(run.run_dir, artifact="latest")
        except FileNotFoundError:
            if run.source_snapshot_dir is None and run.source_run_id is None:
                raise ValueError(
                    "no resumable checkpoint exists for this run yet; "
                    "resume cannot continue it safely"
                ) from None
            run, restored = self._ensure_fork_source_snapshot(run)
            reset_local_clock = True
            self._spawn_worker(run_id=run.id, resume=False)
            message = (
                "training worker relaunched from "
                f"{'rebuilt ' if restored else ''}pinned fork source; "
                f"log: {launching.manager_worker_log_path(run.id)}"
            )
        else:
            self._spawn_worker(run_id=run.id, resume=True)
            message = (
                "training worker resumed from latest checkpoint; "
                f"log: {launching.manager_worker_log_path(run.id)}"
            )

        if reset_local_clock:
            self._store.clear_run_runtime(run.id)
        resumed = self._store.update_run_status(
            run_id=run.id,
            status="running",
            started_at=launching.utc_now() if reset_local_clock else None,
            stopped_at=None,
            message=message,
        )
        if resumed is None:
            raise RuntimeError(f"managed run disappeared during resume: {run.id}")
        return resumed

    def _ensure_fork_source_snapshot(self, run: ManagedRun) -> tuple[ManagedRun, bool]:
        """Restore a missing or incomplete pinned fork source before warm start."""

        snapshot_dir = run.source_snapshot_dir or run_fork_source_dir(run_dir=run.run_dir)
        if run.source_artifact is not None and is_complete_fork_source(
            source_dir=snapshot_dir,
            artifact=run.source_artifact,
        ):
            return run, False
        if run.source_run_id is None or run.source_artifact is None:
            raise ValueError(
                "fork source snapshot is missing and this run cannot rebuild it safely"
            )
        source_run = self._store.get_run(run.source_run_id)
        if source_run is None:
            raise ValueError(f"source run not found for forked run: {run.source_run_id}")
        source_num_timesteps = snapshot_fork_source(
            source_run_dir=source_run.run_dir,
            artifact=run.source_artifact,
            destination_dir=snapshot_dir,
        )
        refreshed = self._store.update_run_fork_source(
            run_id=run.id,
            source_snapshot_dir=snapshot_dir,
            source_num_timesteps=source_num_timesteps,
            lineage_step_offset=source_run.lineage_step_offset + source_num_timesteps,
        )
        if refreshed is None:
            raise RuntimeError(f"managed run disappeared while rebuilding fork source: {run.id}")
        return refreshed, True

    def request_pause(self, *, run_id: str) -> ManagedRun:
        """Request a graceful pause for one running run."""

        return self._request_command(run_id=run_id, command="pause")

    def request_stop(self, *, run_id: str) -> ManagedRun:
        """Request a graceful stop for one running run."""

        return self._request_command(run_id=run_id, command="stop")

    def watch_artifact(
        self,
        *,
        run_id: str,
        artifact: str,
        device: Literal["cpu", "cuda"],
        renderer: Literal["angrylion", "gliden64"] | None,
    ) -> launching.WatchLaunchStatus:
        """Launch the desktop watch app against one saved artifact for one run."""

        return launching.launch_watch_artifact(
            store=self._store,
            run_id=run_id,
            artifact=artifact,
            device=device,
            renderer=renderer,
        )

    def start_career_mode(
        self,
        *,
        save_game_id: str,
        device: Literal["cpu", "cuda"],
        renderer: Literal["angrylion", "gliden64"] | None,
        attempt_seed: int | None,
        deterministic_policy: bool,
        target_kind: str | None = None,
        difficulty: str | None = None,
        cup_id: str | None = None,
        course_id: str | None = None,
    ) -> launching.WatchLaunchStatus:
        """Launch the visible Career Mode runner for one save game."""

        return launching.launch_career_mode_runner(
            store=self._store,
            save_game_id=save_game_id,
            device=device,
            renderer=renderer,
            attempt_seed=attempt_seed,
            deterministic_policy=deterministic_policy,
            target_kind=target_kind,
            difficulty=difficulty,
            cup_id=cup_id,
            course_id=course_id,
        )

    def _request_command(self, *, run_id: str, command: RunCommand) -> ManagedRun:
        run = self._store.get_run(run_id)
        if run is None:
            raise ValueError(f"run not found: {run_id}")
        if run.status != "running":
            raise ValueError("only running runs can be controlled")
        updated = self._store.request_run_command(run_id=run_id, command=command)
        if updated is None:
            raise RuntimeError(f"managed run disappeared during {command}: {run_id}")
        return updated

    def _spawn_worker(self, *, run_id: str, resume: bool) -> None:
        launching.spawn_manager_worker(store=self._store, run_id=run_id, resume=resume)
