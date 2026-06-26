# src/rl_fzerox/core/manager/registry/evaluations/records.py
"""Registry operations for evaluation records, snapshots, and lifecycle."""

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
from rl_fzerox.core.manager.models import (
    ManagedEvaluation,
    ManagedEvaluationPreset,
    ManagedPolicySource,
    PolicySourceKind,
)
from rl_fzerox.core.manager.policy_sources import resolve_policy_source
from rl_fzerox.core.manager.registry.common import new_record_id, utc_now
from rl_fzerox.core.manager.registry.evaluations.presets import get_evaluation_preset
from rl_fzerox.core.manager.run_spec import ManagedRunConfig
from rl_fzerox.core.manager.storage.serialization import config_json

if TYPE_CHECKING:
    from rl_fzerox.core.manager.store import ManagerStore

CANCEL_REQUEST_FILENAME = "cancel.requested"


def create_evaluation(
    store: ManagerStore,
    *,
    name: str,
    source_run_id: str,
    source_artifact: EvaluationCheckpointArtifact,
    policy_mode: EvaluationPolicyMode,
    preset_id: str,
    evaluations_root: Path | None = None,
) -> ManagedEvaluation:
    return create_evaluation_from_policy_source(
        store,
        name=name,
        source_policy_kind="run",
        source_policy_id=source_run_id,
        source_artifact=source_artifact,
        policy_mode=policy_mode,
        preset_id=preset_id,
        evaluations_root=evaluations_root,
    )


def create_evaluation_from_policy_source(
    store: ManagerStore,
    *,
    name: str,
    source_policy_kind: PolicySourceKind,
    source_policy_id: str,
    source_artifact: EvaluationCheckpointArtifact,
    policy_mode: EvaluationPolicyMode,
    preset_id: str,
    evaluations_root: Path | None = None,
) -> ManagedEvaluation:
    """Create one evaluation record and freeze its source checkpoint."""

    preset = get_evaluation_preset(store, preset_id)
    if preset is None:
        raise ValueError("evaluation preset not found")
    preset_version = preset.version

    with store._orm_session() as session:
        policy_source = resolve_policy_source(
            session,
            kind=source_policy_kind,
            source_id=source_policy_id,
            artifact=source_artifact,
            require_policy_artifact=True,
        )
    if policy_source.policy_path is None:
        raise FileNotFoundError("evaluation source policy checkpoint is missing")
    if policy_source.model_path is None:
        raise FileNotFoundError("evaluation source model checkpoint is missing")
    seed = preset.seed
    target = _evaluation_target_for_source(policy_source.config, preset.target)
    config = _evaluation_config_for_source(
        policy_source.config,
        preset=preset,
        target=target,
    )
    source_mtime_ns = policy_source.policy_path.stat().st_mtime_ns
    with store._orm_session() as session:
        existing = evaluation_repository.find_created_evaluation_snapshot(
            session,
            name=name,
            source_policy_kind=source_policy_kind,
            source_policy_id=source_policy_id,
            source_artifact=source_artifact,
            policy_mode=policy_mode,
            seed=seed,
            target=target,
            config=config,
            source_mtime_ns=source_mtime_ns,
            preset_id=preset_id,
            preset_version=preset_version,
        )
        if existing is not None:
            return existing
    created_at = utc_now()
    evaluation_id = new_record_id(name)
    evaluation_dir = (evaluations_root or store.evaluations_root()) / evaluation_id
    checkpoint = snapshot_evaluation_checkpoint(
        _evaluation_checkpoint_source(policy_source),
        destination_dir=evaluation_dir / "checkpoint_snapshot",
    )
    evaluation = ManagedEvaluation(
        id=evaluation_id,
        name=name,
        status="created",
        evaluation_dir=evaluation_dir,
        source_policy_kind=source_policy_kind,
        source_policy_id=source_policy_id,
        source_run_id=source_policy_id if source_policy_kind == "run" else None,
        source_artifact=source_artifact,
        preset_id=preset_id,
        preset_version=preset_version,
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


def _evaluation_checkpoint_source(policy_source: ManagedPolicySource) -> EvaluationCheckpointSource:
    if policy_source.policy_path is None or policy_source.model_path is None:
        raise FileNotFoundError("evaluation source checkpoint is missing")
    return EvaluationCheckpointSource(
        run_id=policy_source.source_run_id,
        run_name=policy_source.source_run_name or policy_source.name,
        run_dir=policy_source.source_dir,
        artifact=policy_source.artifact,
        lineage_step_offset=policy_source.lineage_step_offset,
        policy_path=policy_source.policy_path,
        model_path=policy_source.model_path,
        engine_tuning_state_path=policy_source.engine_tuning_state_path,
        engine_tuning_model_path=policy_source.engine_tuning_model_path,
        local_num_timesteps=policy_source.local_num_timesteps,
        lineage_num_timesteps=policy_source.lineage_num_timesteps,
    )


def delete_evaluation(store: ManagerStore, evaluation_id: str) -> bool:
    """Delete one inactive evaluation snapshot and its artifact directory."""

    store.initialize()
    deleted_at = utc_now()
    with store._orm_session() as session:
        deleted = evaluation_repository.delete_inactive_evaluation(session, evaluation_id)
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
    """Mark one new or failed evaluation as running and expose its summary path."""

    started_at = utc_now()
    cancel_path = evaluation_cancel_request_path(store, evaluation_id)
    try:
        cancel_path.unlink()
    except FileNotFoundError:
        pass
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


def request_evaluation_cancel(
    store: ManagerStore,
    evaluation_id: str,
) -> ManagedEvaluation | None:
    """Request cooperative cancellation of one running evaluation."""

    evaluation = get_evaluation(store, evaluation_id)
    if evaluation is None:
        return None
    if evaluation.status in {"cancelling", "cancelled"}:
        return evaluation
    if evaluation.status != "running":
        raise ValueError(f"only running evaluations can be cancelled, got {evaluation.status}")
    cancel_path = evaluation_cancel_request_path(store, evaluation_id)
    cancel_path.parent.mkdir(parents=True, exist_ok=True)
    cancel_path.write_text(utc_now() + "\n", encoding="utf-8")
    with store._orm_session() as session:
        return evaluation_repository.mark_evaluation_cancelling(
            session,
            evaluation_id=evaluation_id,
            updated_at=utc_now(),
        )


def mark_evaluation_cancelled(store: ManagerStore, evaluation_id: str) -> ManagedEvaluation:
    """Mark one evaluation as cancelled without deleting partial result artifacts."""

    with store._orm_session() as session:
        return evaluation_repository.mark_evaluation_cancelled(
            session,
            evaluation_id=evaluation_id,
            finished_at=utc_now(),
        )


def evaluation_cancel_request_path(store: ManagerStore, evaluation_id: str) -> Path:
    """Return the cooperative cancel marker path for one evaluation."""

    return store.evaluations_root() / evaluation_id / CANCEL_REQUEST_FILENAME


def list_evaluations(store: ManagerStore) -> tuple[ManagedEvaluation, ...]:
    """Return all evaluation records."""

    with store._orm_session() as session:
        return evaluation_repository.list_managed_evaluations(session)


def _evaluation_target_for_source(
    config: ManagedRunConfig,
    target: EvaluationTargetSpec,
) -> EvaluationTargetSpec:
    """Bind policy-owned vehicle selection to one frozen evaluation target."""

    vehicle_ids = target.vehicle_ids or config.vehicle.selected_vehicle_ids
    return EvaluationTargetSpec(
        mode=target.mode,
        course_ids=target.course_ids,
        cup_ids=target.cup_ids,
        difficulties=target.difficulties,
        vehicle_ids=vehicle_ids,
        repeats_per_target=target.repeats_per_target,
        baseline_variant_count=target.baseline_variant_count,
    )


def _evaluation_config_for_source(
    config: ManagedRunConfig,
    *,
    preset: ManagedEvaluationPreset,
    target: EvaluationTargetSpec,
) -> ManagedRunConfig:
    """Project source-run policy config onto the preset-owned evaluation suite."""

    course_ids = target.course_ids or config.tracks.selected_course_ids
    if target.mode == "gp_course":
        race_mode = "gp_race"
        gp_difficulties = target.difficulties or config.tracks.gp_difficulties
        baseline_variant_count = target.baseline_variant_count
    else:
        race_mode = "time_attack"
        gp_difficulties = ()
        baseline_variant_count = 1
    tracks = config.tracks.model_copy(
        update={
            "baseline_variant_count": baseline_variant_count,
            "gp_difficulties": gp_difficulties,
            "include_x_cup": False,
            "race_mode": race_mode,
            "selected_course_ids": course_ids,
        }
    )
    environment = config.environment.model_copy(update={"renderer": preset.renderer})
    return config.model_copy(update={"environment": environment, "tracks": tracks})


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
