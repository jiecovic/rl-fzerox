# src/rl_fzerox/apps/run_manager/launching/run_lifecycle.py
from __future__ import annotations

from pathlib import Path
from typing import Literal, Protocol

from rl_fzerox.apps.run_manager.launching.engine_tuning_source import (
    EngineTuningSourceAction,
    prepare_engine_tuning_fork_source,
)
from rl_fzerox.apps.run_manager.launching.manifest import (
    default_fork_name,
    persist_launch_manifest,
)
from rl_fzerox.apps.run_manager.launching.worker import (
    manager_worker_log_path,
    spawn_manager_worker,
    utc_now,
)
from rl_fzerox.core.manager import ManagedRun, ManagedRunConfig, ManagerStore, new_managed_run_id
from rl_fzerox.core.manager.artifacts.fork_source import (
    clone_fork_source,
    is_complete_fork_source,
    link_or_copy_file,
    load_fork_source_metadata,
    run_fork_source_dir,
    snapshot_fork_source,
)
from rl_fzerox.core.manager.artifacts.paths import predicted_managed_run_dir
from rl_fzerox.core.manager.models import RunCommand
from rl_fzerox.core.manager.run_spec import reset_fork_action_bias_deltas
from rl_fzerox.core.manager.training import (
    assert_managed_fork_compatible,
    build_managed_fork_train_app_config,
    build_managed_fork_train_app_config_from_metadata,
    build_managed_train_app_config,
)
from rl_fzerox.core.training.runs import RUN_LAYOUT, resolve_model_artifact_path
from rl_fzerox.core.training.session.callbacks.track_sampling import TrackSamplingAltBaseline


class WorkerSpawner(Protocol):
    def __call__(self, *, store: ManagerStore, run_id: str, resume: bool) -> None: ...


def launch_run(
    store: ManagerStore,
    *,
    name: str,
    config: ManagedRunConfig,
    draft_id: str | None = None,
    spawn_worker: WorkerSpawner = spawn_manager_worker,
) -> ManagedRun:
    normalized_name = name.strip()
    if not normalized_name:
        raise ValueError("run name is required")

    run_id = new_managed_run_id(normalized_name)
    run_dir = predicted_managed_run_dir(run_id, lineage_id=run_id)
    train_config = build_managed_train_app_config(
        config,
        run_id=run_id,
        run_dir=run_dir,
    )
    run = store.create_run(
        run_id=run_id,
        name=normalized_name,
        config=config,
        explicit_run_dir=run_dir,
        lineage_id=run_id,
        exclude_draft_id=draft_id,
    )
    persist_launch_manifest(run_dir=run.run_dir, train_config=train_config)
    spawn_worker(store=store, run_id=run.id, resume=False)
    launched = store.update_run_status(
        run_id=run.id,
        status="running",
        started_at=utc_now(),
        stopped_at=None,
        message=f"training worker launched; log: {manager_worker_log_path(run.id)}",
    )
    if launched is None:
        raise RuntimeError(f"managed run disappeared during launch: {run.id}")
    return launched


def fork_run(
    store: ManagerStore,
    *,
    run_id: str,
    artifact: Literal["latest", "best"],
    name: str | None = None,
    config: ManagedRunConfig | None = None,
    exclude_draft_id: str | None = None,
    source_snapshot_dir: Path | None = None,
    source_num_timesteps: int | None = None,
    copy_alt_baselines: bool = True,
    engine_tuning_source_action: EngineTuningSourceAction = "convert",
    spawn_worker: WorkerSpawner = spawn_manager_worker,
) -> ManagedRun:
    """Launch one child run warm-started from a parent run checkpoint."""

    source_run = store.get_run(run_id)
    if source_run is None:
        raise ValueError(f"run not found: {run_id}")
    if artifact not in {"latest", "best"}:
        raise ValueError(f"unsupported fork artifact: {artifact}")

    normalized_name = (name or default_fork_name(source_run.name, artifact)).strip()
    if not normalized_name:
        raise ValueError("run name is required")
    child_config = (
        config if config is not None else reset_fork_action_bias_deltas(source_run.config)
    )
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
    prepare_engine_tuning_fork_source(
        config=child_config,
        source_dir=child_source_snapshot_dir,
        artifact=artifact,
        action=engine_tuning_source_action,
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
    child_run = store.create_run(
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
    if copy_alt_baselines:
        _copy_alt_baselines_to_fork(
            store=store,
            source_run=source_run,
            child_run=child_run,
        )
    persist_launch_manifest(run_dir=child_run.run_dir, train_config=train_config)
    spawn_worker(store=store, run_id=child_run.id, resume=False)
    launched = store.update_run_status(
        run_id=child_run.id,
        status="running",
        started_at=utc_now(),
        stopped_at=None,
        message=(
            f"forked from {source_run.name} ({artifact} @ {source_num_timesteps:,} steps); "
            f"log: {manager_worker_log_path(child_run.id)}"
        ),
    )
    if launched is None:
        raise RuntimeError(f"managed child run disappeared during launch: {child_run.id}")
    return launched


def launch_pinned_fork_source(
    store: ManagerStore,
    *,
    name: str,
    config: ManagedRunConfig,
    source_snapshot_dir: Path,
    source_artifact: Literal["latest", "best"],
    source_num_timesteps: int,
    exclude_draft_id: str | None = None,
    engine_tuning_source_action: EngineTuningSourceAction = "convert",
    spawn_worker: WorkerSpawner = spawn_manager_worker,
) -> ManagedRun:
    """Launch one run from a draft-owned immutable fork-source snapshot."""

    normalized_name = name.strip()
    if not normalized_name:
        raise ValueError("run name is required")

    source_metadata = load_fork_source_metadata(source_dir=source_snapshot_dir)
    child_run_id = new_managed_run_id(normalized_name)
    child_run_dir = predicted_managed_run_dir(child_run_id, lineage_id=child_run_id)
    child_source_snapshot_dir = run_fork_source_dir(run_dir=child_run_dir)
    clone_fork_source(
        source_dir=source_snapshot_dir,
        destination_dir=child_source_snapshot_dir,
    )
    prepare_engine_tuning_fork_source(
        config=config,
        source_dir=child_source_snapshot_dir,
        artifact=source_artifact,
        action=engine_tuning_source_action,
    )
    child_lineage_step_offset = (
        source_metadata.source_lineage_num_timesteps or source_num_timesteps
    )
    train_config = build_managed_fork_train_app_config_from_metadata(
        config,
        run_id=child_run_id,
        run_dir=child_run_dir,
        source_run_dir=child_source_snapshot_dir,
        source_artifact=source_artifact,
        source_algorithm=source_metadata.source_algorithm,
        source_auxiliary_state_enabled=(
            source_metadata.source_auxiliary_state_enabled
        ),
        source_auxiliary_state_head_arch=(
            source_metadata.source_auxiliary_state_head_arch
        ),
        tensorboard_step_offset=child_lineage_step_offset,
    )
    child_run = store.create_run(
        run_id=child_run_id,
        name=normalized_name,
        config=config,
        explicit_run_dir=child_run_dir,
        lineage_id=child_run_id,
        lineage_step_offset=child_lineage_step_offset,
        source_artifact=source_artifact,
        source_snapshot_dir=child_source_snapshot_dir,
        source_num_timesteps=source_num_timesteps,
        exclude_draft_id=exclude_draft_id,
    )
    persist_launch_manifest(run_dir=child_run.run_dir, train_config=train_config)
    spawn_worker(store=store, run_id=child_run.id, resume=False)
    launched = store.update_run_status(
        run_id=child_run.id,
        status="running",
        started_at=utc_now(),
        stopped_at=None,
        message=(
            f"forked from pinned checkpoint source "
            f"({source_artifact} @ {source_num_timesteps:,} steps); "
            f"log: {manager_worker_log_path(child_run.id)}"
        ),
    )
    if launched is None:
        raise RuntimeError(f"managed child run disappeared during launch: {child_run.id}")
    return launched


def _copy_alt_baselines_to_fork(
    *,
    store: ManagerStore,
    source_run: ManagedRun,
    child_run: ManagedRun,
) -> None:
    baselines = store.active_run_alt_baselines(source_run.id)
    if not baselines:
        return

    copied_at = utc_now()
    for baseline in baselines:
        child_baseline_id = _forked_alt_baseline_id(
            baseline_id=baseline.id,
            child_run_id=child_run.id,
        )
        child_state_path = (
            child_run.run_dir / RUN_LAYOUT.baselines_dirname / "alt" / f"{child_baseline_id}.state"
        )
        link_or_copy_file(baseline.state_path, child_state_path)
        store.upsert_run_alt_baseline(
            baseline=TrackSamplingAltBaseline(
                id=child_baseline_id,
                run_id=child_run.id,
                course_key=baseline.course_key,
                reset_variant_key=baseline.reset_variant_key,
                source_entry_id=baseline.source_entry_id,
                label=baseline.label,
                state_path=child_state_path,
                weight=baseline.weight,
                enabled=baseline.enabled,
                created_at=copied_at,
                updated_at=copied_at,
            )
        )


def _forked_alt_baseline_id(*, baseline_id: str, child_run_id: str) -> str:
    return f"{baseline_id}-fork-{child_run_id[:8]}"


def resume_run(
    store: ManagerStore,
    *,
    run_id: str,
    spawn_worker: WorkerSpawner = spawn_manager_worker,
) -> ManagedRun:
    """Resume one paused or stopped run in place from its latest checkpoint."""

    run = store.get_run(run_id)
    if run is None:
        raise ValueError(f"run not found: {run_id}")
    if run.status not in {"paused", "stopped", "failed"}:
        raise ValueError("only paused, stopped, or failed runs can be resumed")

    store.clear_run_command(run.id)
    reset_local_clock = False
    try:
        resolve_model_artifact_path(run.run_dir, artifact="latest")
    except FileNotFoundError:
        if run.source_snapshot_dir is None and run.source_run_id is None:
            raise ValueError(
                "no resumable checkpoint exists for this run yet; resume cannot continue it safely"
            ) from None
        run, restored = _ensure_fork_source_snapshot(store, run)
        reset_local_clock = True
        spawn_worker(store=store, run_id=run.id, resume=False)
        message = (
            "training worker relaunched from "
            f"{'rebuilt ' if restored else ''}pinned fork source; "
            f"log: {manager_worker_log_path(run.id)}"
        )
    else:
        spawn_worker(store=store, run_id=run.id, resume=True)
        message = (
            "training worker resumed from latest checkpoint; "
            f"log: {manager_worker_log_path(run.id)}"
        )

    if reset_local_clock:
        store.clear_run_runtime(run.id)
    resumed = store.update_run_status(
        run_id=run.id,
        status="running",
        started_at=utc_now() if reset_local_clock else None,
        stopped_at=None,
        message=message,
    )
    if resumed is None:
        raise RuntimeError(f"managed run disappeared during resume: {run.id}")
    return resumed


def request_run_control(
    store: ManagerStore,
    *,
    run_id: str,
    command: RunCommand,
) -> ManagedRun:
    run = store.get_run(run_id)
    if run is None:
        raise ValueError(f"run not found: {run_id}")
    if run.status != "running":
        raise ValueError("only running runs can be controlled")
    updated = store.request_run_command(run_id=run_id, command=command)
    if updated is None:
        raise RuntimeError(f"managed run disappeared during {command}: {run_id}")
    return updated


def _ensure_fork_source_snapshot(
    store: ManagerStore,
    run: ManagedRun,
) -> tuple[ManagedRun, bool]:
    """Restore a missing or incomplete pinned fork source before warm start."""

    snapshot_dir = run.source_snapshot_dir or run_fork_source_dir(run_dir=run.run_dir)
    if run.source_artifact is not None and is_complete_fork_source(
        source_dir=snapshot_dir,
        artifact=run.source_artifact,
    ):
        return run, False
    if run.source_run_id is None or run.source_artifact is None:
        raise ValueError("fork source snapshot is missing and this run cannot rebuild it safely")
    source_run = store.get_run(run.source_run_id)
    if source_run is None:
        raise ValueError(f"source run not found for forked run: {run.source_run_id}")
    source_num_timesteps = snapshot_fork_source(
        source_run_dir=source_run.run_dir,
        artifact=run.source_artifact,
        destination_dir=snapshot_dir,
    )
    refreshed = store.update_run_fork_source(
        run_id=run.id,
        source_snapshot_dir=snapshot_dir,
        source_num_timesteps=source_num_timesteps,
        lineage_step_offset=source_run.lineage_step_offset + source_num_timesteps,
    )
    if refreshed is None:
        raise RuntimeError(f"managed run disappeared while rebuilding fork source: {run.id}")
    return refreshed, True
