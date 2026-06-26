# src/rl_fzerox/core/manager/registry/lineages/delete.py
"""Deletion flows for managed runs and lineages."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from sqlalchemy import delete, or_, select

from rl_fzerox.core.manager.db.models import (
    LineageGroupModel,
    RunAltBaselineModel,
    RunCommandModel,
    RunDraftModel,
    RunEventModel,
    RunModel,
    RunRuntimeModel,
    RunTrackSamplingArtifactModel,
    RunTrackSamplingEntryModel,
    RunTrackSamplingGeneratedSlotModel,
    RunTrackSamplingRuntimeModel,
    RunWorkerModel,
    SaveGameCourseSetupModel,
)
from rl_fzerox.core.manager.db.repositories.filesystem import queue_delete_tree
from rl_fzerox.core.manager.registry.common import utc_now
from rl_fzerox.core.manager.registry.lineages.order import (
    LineageRunLink,
    delete_order_for_lineage,
)

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from rl_fzerox.core.manager.store import ManagerStore


def delete_run(store: ManagerStore, run_id: str) -> bool:
    """Delete one managed leaf run and queue its filesystem cleanup."""

    store.initialize()
    deleted = False
    deleted_at = utc_now()
    with store._orm_session() as session:
        run = session.get(RunModel, run_id)
        if run is None:
            return False
        if run.status == "running":
            raise ValueError("stop or pause the run before deleting it")
        if session.get(RunCommandModel, run_id) is not None:
            raise ValueError("wait for the pending run command to finish before deleting it")
        if (
            session.scalar(
                select(RunModel.id)
                .where(or_(RunModel.parent_run_id == run_id, RunModel.source_run_id == run_id))
                .limit(1)
            )
            is not None
        ):
            raise ValueError("only leaf runs can be deleted individually")
        if (
            session.scalar(
                select(RunDraftModel.id).where(RunDraftModel.source_run_id == run_id).limit(1)
            )
            is not None
        ):
            raise ValueError("delete or retarget fork drafts that still depend on this run")
        if (
            session.scalar(
                select(SaveGameCourseSetupModel.id)
                .where(SaveGameCourseSetupModel.policy_source_kind == "run")
                .where(SaveGameCourseSetupModel.policy_source_id == run_id)
                .limit(1)
            )
            is not None
        ):
            raise ValueError("remove save-game course setups that still use this policy run")

        run_dir = Path(run.run_dir).expanduser().resolve()
        if run_dir.exists():
            queue_delete_tree(session, path=run_dir, created_at=deleted_at)

        _delete_run_sidecars(session, run_id)
        session.delete(run)
        deleted = True
    store._drain_pending_filesystem_operations()
    return deleted


def delete_lineage(store: ManagerStore, lineage_id: str) -> bool:
    """Delete one full lineage, including its runs and dependent fork drafts."""

    store.initialize()
    deleted_at = utc_now()
    with store._orm_session() as session:
        runs = tuple(
            session.scalars(
                select(RunModel)
                .where(RunModel.lineage_id == lineage_id)
                .order_by(RunModel.created_at.desc(), RunModel.id.desc())
            )
        )
        if not runs:
            return False
        run_ids = tuple(run.id for run in runs)
        for run in runs:
            if run.status == "running":
                raise ValueError("stop all runs in this lineage before deleting it")
        if (
            session.scalar(
                select(RunCommandModel.run_id).where(RunCommandModel.run_id.in_(run_ids)).limit(1)
            )
            is not None
        ):
            raise ValueError("wait for pending run commands to finish before deleting lineage")

        drafts = tuple(
            session.scalars(select(RunDraftModel).where(RunDraftModel.source_run_id.in_(run_ids)))
        )
        run_delete_order = delete_order_for_lineage(
            LineageRunLink(
                run_id=run.id,
                parent_run_id=run.parent_run_id,
                source_run_id=run.source_run_id,
            )
            for run in runs
        )
        lineage_dir = Path(runs[0].run_dir).expanduser().resolve().parent

        for run in runs:
            run_dir = Path(run.run_dir).expanduser().resolve()
            if run_dir.exists():
                queue_delete_tree(session, path=run_dir, created_at=deleted_at)
        for draft in drafts:
            snapshot_dir = draft.source_snapshot_dir
            if isinstance(snapshot_dir, str):
                queue_delete_tree(
                    session,
                    path=Path(snapshot_dir),
                    created_at=deleted_at,
                )
        if lineage_dir.exists():
            queue_delete_tree(session, path=lineage_dir, created_at=deleted_at)

        if drafts:
            draft_ids = tuple(draft.id for draft in drafts)
            session.execute(delete(RunDraftModel).where(RunDraftModel.id.in_(draft_ids)))
        session.execute(delete(LineageGroupModel).where(LineageGroupModel.lineage_id == lineage_id))
        for current_run_id in run_delete_order:
            _delete_run_sidecars(session, current_run_id)
            session.execute(delete(RunModel).where(RunModel.id == current_run_id))
    store._drain_pending_filesystem_operations()
    return True


def _delete_run_sidecars(session: Session, run_id: str) -> None:
    session.execute(delete(RunAltBaselineModel).where(RunAltBaselineModel.run_id == run_id))
    session.execute(
        delete(RunTrackSamplingArtifactModel).where(RunTrackSamplingArtifactModel.run_id == run_id)
    )
    session.execute(
        delete(RunTrackSamplingEntryModel).where(RunTrackSamplingEntryModel.run_id == run_id)
    )
    session.execute(
        delete(RunTrackSamplingGeneratedSlotModel).where(
            RunTrackSamplingGeneratedSlotModel.run_id == run_id
        )
    )
    session.execute(
        delete(RunTrackSamplingRuntimeModel).where(RunTrackSamplingRuntimeModel.run_id == run_id)
    )
    session.execute(delete(RunRuntimeModel).where(RunRuntimeModel.run_id == run_id))
    session.execute(delete(RunCommandModel).where(RunCommandModel.run_id == run_id))
    session.execute(delete(RunWorkerModel).where(RunWorkerModel.run_id == run_id))
    session.execute(delete(RunEventModel).where(RunEventModel.run_id == run_id))
