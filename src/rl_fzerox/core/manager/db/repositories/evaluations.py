# src/rl_fzerox/core/manager/db/repositories/evaluations.py
"""Repository operations for manager-owned evaluation records."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from rl_fzerox.core.evaluation.models import (
    EvaluationCheckpointArtifact,
    EvaluationCheckpointSnapshot,
    EvaluationMode,
    EvaluationPolicyMode,
    EvaluationTargetSpec,
)
from rl_fzerox.core.manager.db.models.evaluations import EvaluationModel
from rl_fzerox.core.manager.models import ManagedEvaluation, ManagedEvaluationStatus


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
            source_run_id=evaluation.source_run_id,
            source_artifact=evaluation.source_artifact,
            policy_mode=evaluation.policy_mode,
            seed=evaluation.seed,
            target_json=json.dumps(asdict(evaluation.target), sort_keys=True),
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


def managed_evaluation_from_model(evaluation: EvaluationModel) -> ManagedEvaluation:
    """Build the public evaluation dataclass from an ORM row."""

    return ManagedEvaluation(
        id=evaluation.id,
        name=evaluation.name,
        status=_evaluation_status(evaluation.status),
        evaluation_dir=Path(evaluation.evaluation_dir),
        source_run_id=evaluation.source_run_id,
        source_artifact=_source_artifact(evaluation.source_artifact),
        policy_mode=_policy_mode(evaluation.policy_mode),
        seed=evaluation.seed,
        target=_target_from_json(evaluation.target_json),
        checkpoint=_checkpoint_from_json(evaluation.checkpoint_json),
        result_json_path=(
            None if evaluation.result_json_path is None else Path(evaluation.result_json_path)
        ),
        error_message=evaluation.error_message,
        created_at=evaluation.created_at,
        updated_at=evaluation.updated_at,
        started_at=evaluation.started_at,
        finished_at=evaluation.finished_at,
    )


def _target_from_json(raw_json: str) -> EvaluationTargetSpec:
    data = json.loads(raw_json)
    if not isinstance(data, dict):
        raise ValueError("evaluation target JSON must be an object")
    return EvaluationTargetSpec(
        mode=_evaluation_mode(data.get("mode")),
        course_ids=tuple(data.get("course_ids", ())),
        cup_ids=tuple(data.get("cup_ids", ())),
        difficulties=tuple(data.get("difficulties", ())),
        vehicle_ids=tuple(data.get("vehicle_ids", ())),
        repeats_per_target=int(data.get("repeats_per_target", 1)),
    )


def _checkpoint_from_json(raw_json: str) -> EvaluationCheckpointSnapshot:
    data = json.loads(raw_json)
    if not isinstance(data, dict):
        raise ValueError("evaluation checkpoint JSON must be an object")
    return EvaluationCheckpointSnapshot(
        source_run_id=data.get("source_run_id"),
        source_run_name=data.get("source_run_name"),
        artifact=_checkpoint_artifact(data.get("artifact")),
        source_policy_path=str(data["source_policy_path"]),
        copied_policy_path=str(data["copied_policy_path"]),
        source_model_path=data.get("source_model_path"),
        copied_model_path=data.get("copied_model_path"),
        local_num_timesteps=data.get("local_num_timesteps"),
        lineage_num_timesteps=data.get("lineage_num_timesteps"),
        source_mtime_ns=data.get("source_mtime_ns"),
    )


def _evaluation_status(value: object) -> ManagedEvaluationStatus:
    match value:
        case "created":
            return "created"
        case "running":
            return "running"
        case "completed":
            return "completed"
        case "failed":
            return "failed"
        case "cancelled":
            return "cancelled"
    raise ValueError(f"Unsupported evaluation status: {value!r}")


def _source_artifact(value: object) -> EvaluationCheckpointArtifact | None:
    if value is None:
        return None
    return _checkpoint_artifact(value)


def _checkpoint_artifact(value: object) -> EvaluationCheckpointArtifact:
    match str(value):
        case "latest":
            return "latest"
        case "best":
            return "best"
        case "final":
            return "final"
    raise ValueError(f"Unsupported evaluation source artifact: {value!r}")


def _policy_mode(value: object) -> EvaluationPolicyMode:
    match str(value):
        case "deterministic":
            return "deterministic"
        case "stochastic":
            return "stochastic"
    raise ValueError(f"Unsupported evaluation policy mode: {value!r}")


def _evaluation_mode(value: object) -> EvaluationMode:
    match str(value):
        case "time_attack":
            return "time_attack"
        case "gp_cup":
            return "gp_cup"
        case "career_target":
            return "career_target"
        case "best_of":
            return "best_of"
    raise ValueError(f"Unsupported evaluation mode: {value!r}")
