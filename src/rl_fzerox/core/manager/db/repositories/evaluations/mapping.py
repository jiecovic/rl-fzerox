# src/rl_fzerox/core/manager/db/repositories/evaluations/mapping.py
"""ORM-to-domain mapping for manager evaluation rows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from rl_fzerox.core.evaluation.models import (
    EvaluationCheckpointArtifact,
    EvaluationCheckpointSnapshot,
    EvaluationMode,
    EvaluationPolicyMode,
    EvaluationTargetSpec,
)
from rl_fzerox.core.manager.db.models.evaluations import (
    EvaluationBaselineSuiteModel,
    EvaluationModel,
    EvaluationPresetModel,
)
from rl_fzerox.core.manager.models import (
    EvaluationBaselineSuiteStatus,
    ManagedEvaluation,
    ManagedEvaluationBaselineSuite,
    ManagedEvaluationPreset,
    ManagedEvaluationStatus,
)
from rl_fzerox.core.manager.storage.serialization import load_config_json


def managed_evaluation_from_model(evaluation: EvaluationModel) -> ManagedEvaluation:
    """Build the public evaluation dataclass from an ORM row."""

    if evaluation.preset_id is None or evaluation.preset_version is None:
        raise ValueError(f"evaluation {evaluation.id!r} has no persisted preset")
    return ManagedEvaluation(
        id=evaluation.id,
        name=evaluation.name,
        status=_evaluation_status(evaluation.status),
        evaluation_dir=Path(evaluation.evaluation_dir),
        source_run_id=evaluation.source_run_id,
        source_artifact=_source_artifact(evaluation.source_artifact),
        preset_id=evaluation.preset_id,
        preset_version=evaluation.preset_version,
        policy_mode=_policy_mode(evaluation.policy_mode),
        seed=evaluation.seed,
        target=_target_from_json(evaluation.target_json),
        config=load_config_json(evaluation.config_json),
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


def evaluation_preset_from_model(preset: EvaluationPresetModel) -> ManagedEvaluationPreset:
    """Build the public evaluation-preset dataclass from an ORM row."""

    return ManagedEvaluationPreset(
        id=preset.id,
        name=preset.name,
        version=preset.version,
        seed=preset.seed,
        renderer=_renderer(preset.renderer),
        target=_target_from_json(preset.target_json),
        builtin=preset.builtin,
        created_at=preset.created_at,
        updated_at=preset.updated_at,
    )


def evaluation_baseline_suite_from_model(
    suite: EvaluationBaselineSuiteModel,
) -> ManagedEvaluationBaselineSuite:
    """Build the public baseline-suite dataclass from an ORM row."""

    return ManagedEvaluationBaselineSuite(
        id=suite.id,
        preset_id=suite.preset_id,
        preset_version=suite.preset_version,
        status=_baseline_suite_status(suite.status),
        suite_dir=Path(suite.suite_dir),
        manifest_path=None if suite.manifest_path is None else Path(suite.manifest_path),
        error_message=suite.error_message,
        created_at=suite.created_at,
        updated_at=suite.updated_at,
        materialized_at=suite.materialized_at,
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
        baseline_variant_count=int(data["baseline_variant_count"]),
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
        case "cancelling":
            return "cancelling"
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
        case "time_attack_course":
            return "time_attack_course"
        case "gp_course":
            return "gp_course"
    raise ValueError(f"Unsupported evaluation mode: {value!r}")


def _renderer(value: object) -> Literal["angrylion", "gliden64"]:
    match str(value):
        case "angrylion":
            return "angrylion"
        case "gliden64":
            return "gliden64"
    raise ValueError(f"Unsupported evaluation renderer: {value!r}")


def _baseline_suite_status(value: object) -> EvaluationBaselineSuiteStatus:
    match str(value):
        case "not_created":
            return "not_created"
        case "ready":
            return "ready"
        case "failed":
            return "failed"
    raise ValueError(f"Unsupported evaluation baseline-suite status: {value!r}")
