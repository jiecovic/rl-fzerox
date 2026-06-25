# src/rl_fzerox/core/manager/transfer/restore.py
"""Restore manager database rows from a validated run bundle manifest."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from sqlalchemy.orm import Session

from rl_fzerox.core.manager.db.models import (
    LineageGroupModel,
    RunModel,
    RunRuntimeModel,
)
from rl_fzerox.core.manager.db.repositories.configs import create_config_snapshot
from rl_fzerox.core.manager.db.repositories.runs import append_run_event, insert_run
from rl_fzerox.core.manager.models import ManagedRun, RunStatus
from rl_fzerox.core.manager.run_spec import ManagedRunConfig
from rl_fzerox.core.manager.transfer.models import (
    RunBundleManifest,
    RunBundleRuntime,
)
from rl_fzerox.core.manager.transfer.rewrite import rewritten_optional_path

if TYPE_CHECKING:
    from rl_fzerox.core.manager.store import ManagerStore


def restore_imported_run_record(
    *,
    store: ManagerStore,
    manifest: RunBundleManifest,
    target_run_id: str,
    target_lineage_id: str,
    target_run_dir: Path,
    replacements: tuple[tuple[str, str], ...],
    archive_name: str,
) -> RunStatus:
    """Insert imported run rows after the payload has been extracted."""

    imported_status = _imported_status(manifest.run.status)
    imported_at = store.utc_now()
    imported_config = ManagedRunConfig.model_validate(manifest.run.config)
    source_snapshot_dir = rewritten_optional_path(
        manifest.run.source_snapshot_dir,
        replacements=replacements,
    )

    store.initialize()
    with store._orm_session() as session:
        parent_run_id = _existing_run_id(session, manifest.run.parent_run_id)
        source_run_id = _existing_run_id(session, manifest.run.source_run_id)
        config_snapshot = create_config_snapshot(
            session,
            kind="import",
            config=imported_config,
            created_at=imported_at,
        )
        imported_run = ManagedRun(
            id=target_run_id,
            name=manifest.run.name,
            status=imported_status,
            config=imported_config,
            config_hash=config_snapshot.config_hash,
            run_dir=target_run_dir,
            lineage_id=target_lineage_id,
            lineage_step_offset=manifest.run.lineage_step_offset,
            parent_run_id=parent_run_id,
            source_run_id=source_run_id,
            source_artifact=manifest.run.source_artifact if source_run_id is not None else None,
            source_snapshot_dir=None if source_snapshot_dir is None else Path(source_snapshot_dir),
            source_num_timesteps=(
                manifest.run.source_num_timesteps if source_run_id is not None else None
            ),
            created_at=manifest.run.created_at,
            started_at=manifest.run.started_at,
            stopped_at=(
                manifest.run.stopped_at if imported_status == manifest.run.status else imported_at
            ),
        )
        insert_run(session, run=imported_run, config_snapshot_id=config_snapshot.id)
        session.flush()
        for group_name in manifest.run.lineage_groups:
            session.merge(
                LineageGroupModel(
                    lineage_id=target_lineage_id,
                    group_name=group_name,
                    updated_at=imported_at,
                )
            )
        _insert_runtime(session, run_id=target_run_id, runtime=manifest.run.runtime)
        for event in manifest.run.events:
            append_run_event(
                session,
                run_id=target_run_id,
                created_at=event.created_at,
                kind=event.kind,
                message=event.message,
            )
        append_run_event(
            session,
            run_id=target_run_id,
            created_at=imported_at,
            kind="imported",
            message=f"run imported from {archive_name}",
        )
    return imported_status


def _imported_status(status: RunStatus) -> RunStatus:
    match status:
        case "running" | "paused" | "created":
            return "stopped"
        case "archived":
            return "archived"
        case "stopped" | "finished" | "failed":
            return status


def _existing_run_id(session: Session, run_id: str | None) -> str | None:
    if run_id is None:
        return None
    run = session.get(RunModel, run_id)
    return None if run is None else run.id


def _insert_runtime(
    session: Session,
    *,
    run_id: str,
    runtime: RunBundleRuntime | None,
) -> None:
    if runtime is None:
        return
    session.add(
        RunRuntimeModel(
            run_id=run_id,
            total_timesteps=runtime.total_timesteps,
            num_timesteps=runtime.num_timesteps,
            progress_fraction=runtime.progress_fraction,
            updated_at=runtime.updated_at,
            fps=runtime.fps,
            episode_reward_mean=runtime.episode_reward_mean,
            episode_length_mean=runtime.episode_length_mean,
            approx_kl=runtime.approx_kl,
            entropy_loss=runtime.entropy_loss,
            value_loss=runtime.value_loss,
            policy_gradient_loss=runtime.policy_gradient_loss,
        )
    )
