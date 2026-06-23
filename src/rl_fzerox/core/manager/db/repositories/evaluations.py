# src/rl_fzerox/core/manager/db/repositories/evaluations.py
"""Repository operations for manager-owned evaluation records."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Literal

from sqlalchemy import select
from sqlalchemy.orm import Session

from rl_fzerox.core.domain.courses import BUILT_IN_COURSES
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
from rl_fzerox.core.manager.run_spec import ManagedRunConfig
from rl_fzerox.core.manager.storage.serialization import config_json, load_config_json

_DEFAULT_EVALUATION_SEED = 2_262_218_583
_DEFAULT_EVALUATION_REPEATS = 10


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
    source_run_id: str,
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
            .where(EvaluationModel.source_run_id == source_run_id)
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


def upsert_default_evaluation_presets(session: Session, *, now: str) -> None:
    """Refresh built-in evaluation presets from the current code definition."""

    for preset in _default_evaluation_presets(now=now):
        existing = session.get(EvaluationPresetModel, preset.id)
        if existing is None:
            session.add(_preset_model_from_dataclass(preset))
            continue
        if not existing.builtin:
            continue
        _update_preset_model(existing, preset)


def list_evaluation_presets(session: Session) -> tuple[ManagedEvaluationPreset, ...]:
    """Return evaluation presets in manager display order."""

    presets = tuple(
        session.scalars(
            select(EvaluationPresetModel).order_by(
                EvaluationPresetModel.builtin.desc(),
                EvaluationPresetModel.name.asc(),
                EvaluationPresetModel.id.asc(),
            )
        )
    )
    return tuple(evaluation_preset_from_model(preset) for preset in presets)


def get_evaluation_preset(
    session: Session,
    preset_id: str,
) -> ManagedEvaluationPreset | None:
    """Return one evaluation preset by stable id."""

    preset = session.get(EvaluationPresetModel, preset_id)
    return None if preset is None else evaluation_preset_from_model(preset)


def insert_evaluation_preset(
    session: Session,
    *,
    preset: ManagedEvaluationPreset,
) -> ManagedEvaluationPreset:
    """Insert one custom evaluation preset."""

    session.add(_preset_model_from_dataclass(preset))
    session.flush()
    return preset


def delete_evaluation_preset(
    session: Session,
    preset_id: str,
) -> tuple[ManagedEvaluationPreset, tuple[ManagedEvaluationBaselineSuite, ...]] | None:
    """Delete one unused custom preset and return its baseline-suite rows."""

    preset = session.get(EvaluationPresetModel, preset_id)
    if preset is None:
        return None
    managed = evaluation_preset_from_model(preset)
    if managed.builtin:
        raise ValueError("built-in evaluation presets cannot be deleted")
    referenced_evaluation_id = session.scalar(
        select(EvaluationModel.id).where(EvaluationModel.preset_id == preset_id).limit(1)
    )
    if referenced_evaluation_id is not None:
        raise ValueError("evaluation preset is referenced by evaluation records")
    suites = tuple(
        session.scalars(
            select(EvaluationBaselineSuiteModel).where(
                EvaluationBaselineSuiteModel.preset_id == preset_id
            )
        )
    )
    managed_suites = tuple(evaluation_baseline_suite_from_model(suite) for suite in suites)
    for suite in suites:
        session.delete(suite)
    session.delete(preset)
    return managed, managed_suites


def ensure_evaluation_baseline_suite(
    session: Session,
    *,
    suite: ManagedEvaluationBaselineSuite,
) -> ManagedEvaluationBaselineSuite:
    """Ensure one preset-version baseline-suite row exists."""

    existing = session.get(EvaluationBaselineSuiteModel, suite.id)
    if existing is None:
        session.add(
            EvaluationBaselineSuiteModel(
                id=suite.id,
                preset_id=suite.preset_id,
                preset_version=suite.preset_version,
                status=suite.status,
                suite_dir=str(suite.suite_dir),
                manifest_path=None if suite.manifest_path is None else str(suite.manifest_path),
                error_message=suite.error_message,
                created_at=suite.created_at or "",
                updated_at=suite.updated_at or "",
                materialized_at=suite.materialized_at,
            )
        )
        session.flush()
        return suite
    return evaluation_baseline_suite_from_model(existing)


def upsert_evaluation_baseline_suite_status(
    session: Session,
    *,
    suite: ManagedEvaluationBaselineSuite,
) -> ManagedEvaluationBaselineSuite:
    """Persist the latest known filesystem status for one baseline suite."""

    existing = session.get(EvaluationBaselineSuiteModel, suite.id)
    if existing is None:
        return ensure_evaluation_baseline_suite(session, suite=suite)
    existing.status = suite.status
    existing.suite_dir = str(suite.suite_dir)
    existing.manifest_path = None if suite.manifest_path is None else str(suite.manifest_path)
    existing.error_message = suite.error_message
    existing.updated_at = suite.updated_at or existing.updated_at
    existing.materialized_at = suite.materialized_at
    session.flush()
    return evaluation_baseline_suite_from_model(existing)


def list_evaluation_baseline_suites(
    session: Session,
) -> tuple[ManagedEvaluationBaselineSuite, ...]:
    """Return known evaluation baseline suites."""

    suites = tuple(
        session.scalars(
            select(EvaluationBaselineSuiteModel).order_by(
                EvaluationBaselineSuiteModel.preset_id.asc(),
                EvaluationBaselineSuiteModel.preset_version.asc(),
            )
        )
    )
    return tuple(evaluation_baseline_suite_from_model(suite) for suite in suites)


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


def _required_evaluation(session: Session, evaluation_id: str) -> EvaluationModel:
    evaluation = session.get(EvaluationModel, evaluation_id)
    if evaluation is None:
        raise ValueError(f"evaluation not found: {evaluation_id}")
    return evaluation


def _preset_model_from_dataclass(preset: ManagedEvaluationPreset) -> EvaluationPresetModel:
    return EvaluationPresetModel(
        id=preset.id,
        name=preset.name,
        version=preset.version,
        seed=preset.seed,
        renderer=preset.renderer,
        target_json=json.dumps(asdict(preset.target), sort_keys=True),
        builtin=preset.builtin,
        created_at=preset.created_at,
        updated_at=preset.updated_at,
    )


def _update_preset_model(
    model: EvaluationPresetModel,
    preset: ManagedEvaluationPreset,
) -> None:
    model.name = preset.name
    model.version = preset.version
    model.seed = preset.seed
    model.renderer = preset.renderer
    model.target_json = json.dumps(asdict(preset.target), sort_keys=True)
    model.builtin = preset.builtin
    model.updated_at = preset.updated_at


def _default_evaluation_presets(*, now: str) -> tuple[ManagedEvaluationPreset, ...]:
    all_course_ids = tuple(course.id for course in BUILT_IN_COURSES)
    return (
        ManagedEvaluationPreset(
            id="time_attack_all_courses",
            name="Time Attack course · all courses",
            version=1,
            seed=_DEFAULT_EVALUATION_SEED,
            renderer="gliden64",
            target=EvaluationTargetSpec(
                mode="time_attack_course",
                course_ids=all_course_ids,
                repeats_per_target=_DEFAULT_EVALUATION_REPEATS,
            ),
            builtin=True,
            created_at=now,
            updated_at=now,
        ),
        ManagedEvaluationPreset(
            id="gp_course_master_all_courses",
            name="GP course · Master · all courses",
            version=1,
            seed=_DEFAULT_EVALUATION_SEED,
            renderer="gliden64",
            target=EvaluationTargetSpec(
                mode="gp_course",
                course_ids=all_course_ids,
                difficulties=("master",),
                repeats_per_target=_DEFAULT_EVALUATION_REPEATS,
            ),
            builtin=True,
            created_at=now,
            updated_at=now,
        ),
    )
