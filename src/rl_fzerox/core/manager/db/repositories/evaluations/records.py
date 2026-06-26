# src/rl_fzerox/core/manager/db/repositories/evaluations/records.py
"""Repository operations for evaluation records and lifecycle state."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from rl_fzerox.core.evaluation.models import (
    EvaluationCheckpointArtifact,
    EvaluationPolicyMode,
    EvaluationTargetSpec,
)
from rl_fzerox.core.manager.db.models.evaluations import EvaluationModel
from rl_fzerox.core.manager.db.repositories.evaluations.mapping import (
    managed_evaluation_from_model,
)
from rl_fzerox.core.manager.models import ManagedEvaluation
from rl_fzerox.core.manager.run_spec import ManagedRunConfig
from rl_fzerox.core.manager.storage.serialization import config_json


def insert_evaluation(
    session: Session,
    *,
    evaluation: ManagedEvaluation,
) -> None:
    """Insert one evaluation record."""

    session.add(
        EvaluationModel(
            id=evaluation.id,
            name=evaluation.name,
            status=evaluation.status,
            evaluation_dir=str(evaluation.evaluation_dir),
            source_policy_kind=evaluation.source_policy_kind,
            source_policy_id=evaluation.source_policy_id,
            source_run_id=evaluation.source_run_id,
            source_artifact=evaluation.source_artifact,
            preset_id=evaluation.preset_id,
            preset_version=evaluation.preset_version,
            policy_mode=evaluation.policy_mode,
            seed=evaluation.seed,
            target_json=json.dumps(asdict(evaluation.target), sort_keys=True),
            config_json=config_json(evaluation.config),
            checkpoint_json=json.dumps(asdict(evaluation.checkpoint), sort_keys=True),
            result_json_path=(
                None if evaluation.result_json_path is None else str(evaluation.result_json_path)
            ),
            error_message=evaluation.error_message,
            created_at=evaluation.created_at,
            updated_at=evaluation.updated_at,
            started_at=evaluation.started_at,
            finished_at=evaluation.finished_at,
        )
    )


def get_managed_evaluation(
    session: Session,
    evaluation_id: str,
) -> ManagedEvaluation | None:
    """Return one manager-owned evaluation by id."""

    evaluation = session.get(EvaluationModel, evaluation_id)
    return None if evaluation is None else managed_evaluation_from_model(evaluation)


def list_managed_evaluations(session: Session) -> tuple[ManagedEvaluation, ...]:
    """Return evaluations in manager display order."""

    evaluations = tuple(
        session.scalars(
            select(EvaluationModel).order_by(
                EvaluationModel.created_at.desc(),
                EvaluationModel.id.desc(),
            )
        )
    )
    return tuple(managed_evaluation_from_model(evaluation) for evaluation in evaluations)


def find_created_evaluation_snapshot(
    session: Session,
    *,
    name: str,
    source_policy_kind: str,
    source_policy_id: str | None,
    source_artifact: EvaluationCheckpointArtifact,
    policy_mode: EvaluationPolicyMode,
    seed: int,
    target: EvaluationTargetSpec,
    config: ManagedRunConfig,
    source_mtime_ns: int,
    preset_id: str,
    preset_version: int,
) -> ManagedEvaluation | None:
    """Return an existing identical created snapshot, if one exists."""

    candidates = tuple(
        session.scalars(
            select(EvaluationModel)
            .where(EvaluationModel.name == name)
            .where(EvaluationModel.status == "created")
            .where(EvaluationModel.source_policy_kind == source_policy_kind)
            .where(EvaluationModel.source_policy_id == source_policy_id)
            .where(EvaluationModel.source_artifact == source_artifact)
            .where(EvaluationModel.policy_mode == policy_mode)
            .where(EvaluationModel.seed == seed)
            .where(EvaluationModel.preset_id == preset_id)
            .where(EvaluationModel.preset_version == preset_version)
            .order_by(EvaluationModel.created_at.desc(), EvaluationModel.id.desc())
        )
    )
    for candidate in candidates:
        evaluation = managed_evaluation_from_model(candidate)
        if (
            evaluation.target == target
            and evaluation.config == config
            and evaluation.checkpoint.source_mtime_ns == source_mtime_ns
        ):
            return evaluation
    return None


def delete_inactive_evaluation(session: Session, evaluation_id: str) -> ManagedEvaluation | None:
    """Delete one evaluation record that has no active worker."""

    evaluation = session.get(EvaluationModel, evaluation_id)
    if evaluation is None:
        return None
    managed = managed_evaluation_from_model(evaluation)
    if managed.status in {"running", "cancelling"}:
        raise ValueError("active evaluation snapshots cannot be deleted")
    session.delete(evaluation)
    return managed


def update_evaluation_name(
    session: Session,
    *,
    evaluation_id: str,
    name: str,
    updated_at: str,
) -> ManagedEvaluation | None:
    """Rename one manager-owned evaluation."""

    evaluation = session.get(EvaluationModel, evaluation_id)
    if evaluation is None:
        return None
    evaluation.name = name
    evaluation.updated_at = updated_at
    session.flush()
    return managed_evaluation_from_model(evaluation)


def mark_evaluation_running(
    session: Session,
    *,
    evaluation_id: str,
    started_at: str,
    result_json_path: Path,
) -> ManagedEvaluation:
    """Mark one new or failed evaluation as running."""

    evaluation = _required_evaluation(session, evaluation_id)
    managed = managed_evaluation_from_model(evaluation)
    if managed.status not in {"created", "failed", "cancelled"}:
        raise ValueError(
            f"evaluation must be created, failed, or cancelled before start, got {managed.status!r}"
        )
    evaluation.status = "running"
    evaluation.started_at = started_at
    evaluation.finished_at = None
    evaluation.updated_at = started_at
    evaluation.result_json_path = str(result_json_path)
    evaluation.error_message = None
    session.flush()
    return managed_evaluation_from_model(evaluation)


def mark_evaluation_completed(
    session: Session,
    *,
    evaluation_id: str,
    finished_at: str,
) -> ManagedEvaluation:
    """Mark one running evaluation as completed."""

    evaluation = _required_evaluation(session, evaluation_id)
    managed = managed_evaluation_from_model(evaluation)
    if managed.status == "cancelled":
        return managed
    if managed.status not in {"running", "cancelling"}:
        raise ValueError(
            f"evaluation must be running or cancelling before completion, got {managed.status!r}"
        )
    evaluation.status = "completed"
    evaluation.finished_at = finished_at
    evaluation.updated_at = finished_at
    evaluation.error_message = None
    session.flush()
    return managed_evaluation_from_model(evaluation)


def mark_evaluation_failed(
    session: Session,
    *,
    evaluation_id: str,
    finished_at: str,
    error_message: str,
) -> ManagedEvaluation:
    """Mark one evaluation as failed with a user-facing error."""

    evaluation = _required_evaluation(session, evaluation_id)
    managed = managed_evaluation_from_model(evaluation)
    if managed.status == "cancelled":
        return managed
    evaluation.status = "failed"
    evaluation.finished_at = finished_at
    evaluation.updated_at = finished_at
    evaluation.error_message = error_message
    session.flush()
    return managed_evaluation_from_model(evaluation)


def mark_evaluation_cancelling(
    session: Session,
    *,
    evaluation_id: str,
    updated_at: str,
) -> ManagedEvaluation:
    """Record a cooperative cancel request without claiming the worker has exited."""

    evaluation = _required_evaluation(session, evaluation_id)
    managed = managed_evaluation_from_model(evaluation)
    if managed.status == "cancelled":
        return managed
    if managed.status == "cancelling":
        return managed
    if managed.status != "running":
        raise ValueError(f"only running evaluations can be cancelled, got {managed.status}")
    evaluation.status = "cancelling"
    evaluation.updated_at = updated_at
    evaluation.error_message = None
    session.flush()
    return managed_evaluation_from_model(evaluation)


def mark_evaluation_cancelled(
    session: Session,
    *,
    evaluation_id: str,
    finished_at: str,
) -> ManagedEvaluation:
    """Mark one evaluation as cancelled."""

    evaluation = _required_evaluation(session, evaluation_id)
    managed = managed_evaluation_from_model(evaluation)
    if managed.status == "cancelled":
        return managed
    if managed.status not in {"running", "cancelling"}:
        raise ValueError(
            f"evaluation must be running or cancelling before cancel, got {managed.status!r}"
        )
    evaluation.status = "cancelled"
    evaluation.finished_at = finished_at
    evaluation.updated_at = finished_at
    evaluation.error_message = None
    session.flush()
    return managed_evaluation_from_model(evaluation)


def _required_evaluation(session: Session, evaluation_id: str) -> EvaluationModel:
    evaluation = session.get(EvaluationModel, evaluation_id)
    if evaluation is None:
        raise ValueError(f"evaluation not found: {evaluation_id}")
    return evaluation
