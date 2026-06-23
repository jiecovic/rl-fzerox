# src/rl_fzerox/core/manager/registry/evaluations.py
"""Manager registry operations for evaluation records and artifacts."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

from rl_fzerox.core.evaluation.models import (
    EvaluationCheckpointArtifact,
    EvaluationPolicyMode,
    EvaluationSpec,
    EvaluationTargetSpec,
)
from rl_fzerox.core.evaluation.snapshots import (
    EvaluationCheckpointSource,
    snapshot_evaluation_checkpoint,
)
from rl_fzerox.core.manager.db.repositories import evaluations as evaluation_repository
from rl_fzerox.core.manager.db.repositories.filesystem import queue_delete_tree
from rl_fzerox.core.manager.models import ManagedEvaluation
from rl_fzerox.core.manager.registry.common import new_record_id, utc_now
from rl_fzerox.core.manager.run_spec import ManagedRunConfig
from rl_fzerox.core.manager.storage.serialization import config_json
from rl_fzerox.core.training.runs import resolve_policy_artifact_path

if TYPE_CHECKING:
    from rl_fzerox.core.manager.store import ManagerStore


def create_evaluation(
    store: ManagerStore,
    *,
    name: str,
    source_run_id: str,
    source_artifact: EvaluationCheckpointArtifact,
    policy_mode: EvaluationPolicyMode,
    seed: int,
    target: EvaluationTargetSpec,
    config: ManagedRunConfig,
    evaluations_root: Path | None = None,
) -> ManagedEvaluation:
    """Create one evaluation record and freeze its source checkpoint."""

    source_run = store.get_run(source_run_id)
    if source_run is None:
        raise ValueError("source run not found")
    source_policy_path = resolve_policy_artifact_path(source_run.run_dir, artifact=source_artifact)
    source_mtime_ns = source_policy_path.stat().st_mtime_ns
    with store._orm_session() as session:
        existing = evaluation_repository.find_created_evaluation_snapshot(
            session,
            name=name,
            source_run_id=source_run.id,
            source_artifact=source_artifact,
            policy_mode=policy_mode,
            seed=seed,
            target=target,
            config=config,
            source_mtime_ns=source_mtime_ns,
        )
        if existing is not None:
            return existing
    created_at = utc_now()
    evaluation_id = new_record_id(name)
    evaluation_dir = (evaluations_root or store.evaluations_root()) / evaluation_id
    checkpoint = snapshot_evaluation_checkpoint(
        EvaluationCheckpointSource(
            run_id=source_run.id,
            run_name=source_run.name,
            run_dir=source_run.run_dir,
            artifact=source_artifact,
            lineage_step_offset=source_run.lineage_step_offset,
        ),
        destination_dir=evaluation_dir / "checkpoint_snapshot",
    )
    evaluation = ManagedEvaluation(
        id=evaluation_id,
        name=name,
        status="created",
        evaluation_dir=evaluation_dir,
        source_run_id=source_run.id,
        source_artifact=source_artifact,
        policy_mode=policy_mode,
        seed=seed,
        target=target,
        config=config,
        checkpoint=checkpoint,
        created_at=created_at,
        updated_at=created_at,
    )
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    _write_evaluation_spec(evaluation)
    with store._orm_session() as session:
        evaluation_repository.insert_evaluation(session, evaluation=evaluation)
    return evaluation


def delete_evaluation(store: ManagerStore, evaluation_id: str) -> bool:
    """Delete one created evaluation snapshot and its artifact directory."""

    store.initialize()
    deleted_at = utc_now()
    with store._orm_session() as session:
        deleted = evaluation_repository.delete_created_evaluation(session, evaluation_id)
        if deleted is None:
            return False
        queue_delete_tree(session, path=deleted.evaluation_dir, created_at=deleted_at)
    store._drain_pending_filesystem_operations()
    return True


def update_evaluation_name(
    store: ManagerStore,
    *,
    evaluation_id: str,
    name: str,
) -> ManagedEvaluation | None:
    """Rename one evaluation record."""

    with store._orm_session() as session:
        return evaluation_repository.update_evaluation_name(
            session,
            evaluation_id=evaluation_id,
            name=name,
            updated_at=utc_now(),
        )


def get_evaluation(store: ManagerStore, evaluation_id: str) -> ManagedEvaluation | None:
    """Return one evaluation record by id."""

    with store._orm_session() as session:
        return evaluation_repository.get_managed_evaluation(session, evaluation_id)


def mark_evaluation_running(store: ManagerStore, evaluation_id: str) -> ManagedEvaluation:
    """Mark one evaluation as running and expose its summary path."""

    started_at = utc_now()
    with store._orm_session() as session:
        evaluation = evaluation_repository.mark_evaluation_running(
            session,
            evaluation_id=evaluation_id,
            started_at=started_at,
            result_json_path=store.evaluations_root() / evaluation_id / "evaluation.summary.json",
        )
    return evaluation


def mark_evaluation_completed(store: ManagerStore, evaluation_id: str) -> ManagedEvaluation:
    """Mark one evaluation as completed."""

    with store._orm_session() as session:
        return evaluation_repository.mark_evaluation_completed(
            session,
            evaluation_id=evaluation_id,
            finished_at=utc_now(),
        )


def mark_evaluation_failed(
    store: ManagerStore,
    evaluation_id: str,
    *,
    error_message: str,
) -> ManagedEvaluation:
    """Mark one evaluation as failed."""

    with store._orm_session() as session:
        return evaluation_repository.mark_evaluation_failed(
            session,
            evaluation_id=evaluation_id,
            finished_at=utc_now(),
            error_message=error_message,
        )


def list_evaluations(store: ManagerStore) -> tuple[ManagedEvaluation, ...]:
    """Return all evaluation records."""

    with store._orm_session() as session:
        return evaluation_repository.list_managed_evaluations(session)


def _write_evaluation_spec(evaluation: ManagedEvaluation) -> None:
    spec = EvaluationSpec(
        evaluation_id=evaluation.id,
        seed=evaluation.seed,
        target=evaluation.target,
        checkpoint=evaluation.checkpoint,
        policy_mode=evaluation.policy_mode,
    )
    payload = asdict(spec)
    path = evaluation.evaluation_dir / "evaluation.spec.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    config_path = evaluation.evaluation_dir / "evaluation.config.json"
    config_path.write_text(config_json(evaluation.config) + "\n", encoding="utf-8")
