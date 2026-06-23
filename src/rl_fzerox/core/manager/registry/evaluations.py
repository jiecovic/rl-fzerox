# src/rl_fzerox/core/manager/registry/evaluations.py
"""Manager registry operations for evaluation records and artifacts."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Literal

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
    ManagedEvaluationBaselineSuite,
    ManagedEvaluationPreset,
)
from rl_fzerox.core.manager.registry.common import new_record_id, utc_now
from rl_fzerox.core.manager.run_spec import ManagedRunConfig
from rl_fzerox.core.manager.storage.serialization import config_json
from rl_fzerox.core.training.runs import RUN_LAYOUT, resolve_policy_artifact_path

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
    device: Literal["cpu", "cuda"] = "cuda",
    evaluations_root: Path | None = None,
) -> ManagedEvaluation:
    """Create one evaluation record and freeze its source checkpoint."""

    preset = get_evaluation_preset(store, preset_id)
    if preset is None:
        raise ValueError("evaluation preset not found")
    preset_version = preset.version

    source_run = store.get_run(source_run_id)
    if source_run is None:
        raise ValueError("source run not found")
    seed = preset.seed
    target = _evaluation_target_for_source(source_run.config, preset.target)
    config = _evaluation_config_for_source(
        source_run.config,
        preset=preset,
        target=target,
        device=device,
    )
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
            preset_id=preset_id,
            preset_version=preset_version,
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
    )


def _evaluation_config_for_source(
    config: ManagedRunConfig,
    *,
    preset: ManagedEvaluationPreset,
    target: EvaluationTargetSpec,
    device: Literal["cpu", "cuda"],
) -> ManagedRunConfig:
    """Project source-run policy config onto the preset-owned evaluation suite."""

    course_ids = target.course_ids or config.tracks.selected_course_ids
    if target.mode == "gp_course":
        race_mode = "gp_race"
        gp_difficulties = target.difficulties or config.tracks.gp_difficulties
    else:
        race_mode = "time_attack"
        gp_difficulties = ()
    tracks = config.tracks.model_copy(
        update={
            "baseline_variant_count": 1,
            "gp_difficulties": gp_difficulties,
            "include_x_cup": False,
            "race_mode": race_mode,
            "selected_course_ids": course_ids,
        }
    )
    environment = config.environment.model_copy(update={"renderer": preset.renderer})
    train = config.train.model_copy(update={"device": device})
    return config.model_copy(update={"environment": environment, "tracks": tracks, "train": train})


def get_evaluation_preset(
    store: ManagerStore,
    preset_id: str,
) -> ManagedEvaluationPreset | None:
    """Return one persisted evaluation preset."""

    store.initialize()
    with store._orm_session() as session:
        return evaluation_repository.get_evaluation_preset(session, preset_id)


def list_evaluation_presets(store: ManagerStore) -> tuple[ManagedEvaluationPreset, ...]:
    """Return persisted evaluation presets."""

    store.initialize()
    with store._orm_session() as session:
        return evaluation_repository.list_evaluation_presets(session)


def create_evaluation_preset(
    store: ManagerStore,
    *,
    name: str,
    seed: int,
    renderer: Literal["angrylion", "gliden64"],
    target: EvaluationTargetSpec,
) -> ManagedEvaluationPreset:
    """Create one immutable custom benchmark preset."""

    store.initialize()
    _validate_preset_target(target)
    created_at = utc_now()
    preset = ManagedEvaluationPreset(
        id=new_record_id(name),
        name=name,
        version=1,
        seed=seed,
        renderer=renderer,
        target=target,
        builtin=False,
        created_at=created_at,
        updated_at=created_at,
    )
    with store._orm_session() as session:
        return evaluation_repository.insert_evaluation_preset(session, preset=preset)


def delete_evaluation_preset(store: ManagerStore, preset_id: str) -> bool:
    """Delete one unused custom preset and its materialized baseline suite."""

    store.initialize()
    deleted_at = utc_now()
    with store._orm_session() as session:
        deleted = evaluation_repository.delete_evaluation_preset(session, preset_id)
        if deleted is None:
            return False
        preset, suites = deleted
        if suites:
            for suite in suites:
                queue_delete_tree(session, path=suite.suite_dir, created_at=deleted_at)
        else:
            suite = _baseline_suite_for_preset(
                store,
                preset_id=preset.id,
                preset_version=preset.version,
                created_at=None,
            )
            queue_delete_tree(session, path=suite.suite_dir, created_at=deleted_at)
    store._drain_pending_filesystem_operations()
    return True


def list_evaluation_baseline_suites(
    store: ManagerStore,
) -> tuple[ManagedEvaluationBaselineSuite, ...]:
    """Return materialization status for each preset without creating DB rows."""

    store.initialize()
    now = utc_now()
    with store._orm_session() as session:
        presets = evaluation_repository.list_evaluation_presets(session)
        persisted_suites = {
            (suite.preset_id, suite.preset_version): suite
            for suite in evaluation_repository.list_evaluation_baseline_suites(session)
        }
    return tuple(
        _with_filesystem_suite_status(
            persisted_suites.get(
                (preset.id, preset.version),
                _baseline_suite_for_preset(
                    store,
                    preset_id=preset.id,
                    preset_version=preset.version,
                    created_at=None,
                ),
            ),
            updated_at=now,
        )
        for preset in presets
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
    if evaluation.status == "cancelled":
        return evaluation
    if evaluation.status != "running":
        raise ValueError(f"only running evaluations can be cancelled, got {evaluation.status}")
    cancel_path = evaluation_cancel_request_path(store, evaluation_id)
    cancel_path.parent.mkdir(parents=True, exist_ok=True)
    cancel_path.write_text(utc_now() + "\n", encoding="utf-8")
    return mark_evaluation_cancelled(store, evaluation_id)


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


def _validate_preset_target(target: EvaluationTargetSpec) -> None:
    if target.mode == "gp_course" and len(target.difficulties) != 1:
        raise ValueError("gp_course evaluation presets require exactly one difficulty")
    if target.mode == "time_attack_course" and target.difficulties:
        raise ValueError("time_attack_course evaluation presets must not set difficulties")


def baseline_suite_for_evaluation(evaluation: ManagedEvaluation) -> ManagedEvaluationBaselineSuite:
    """Return the suite row implied by one preset-backed evaluation."""

    return _with_filesystem_suite_status(
        _baseline_suite_for_preset(
            None,
            preset_id=evaluation.preset_id,
            preset_version=evaluation.preset_version,
            created_at=None,
            evaluations_root=evaluation.evaluation_dir.parent,
        )
    )


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


def _baseline_suite_for_preset(
    store: ManagerStore | None,
    *,
    preset_id: str,
    preset_version: int,
    created_at: str | None,
    evaluations_root: Path | None = None,
) -> ManagedEvaluationBaselineSuite:
    if evaluations_root is None:
        if store is None:
            raise ValueError("store or evaluations_root is required")
        root = store.evaluations_root()
    else:
        root = evaluations_root
    suite_id = f"{preset_id}-v{preset_version}"
    suite_dir = root / "_baseline_suites" / suite_id
    return ManagedEvaluationBaselineSuite(
        id=suite_id,
        preset_id=preset_id,
        preset_version=preset_version,
        status="not_created",
        suite_dir=suite_dir,
        manifest_path=suite_dir / RUN_LAYOUT.config_filename,
        created_at=created_at,
        updated_at=created_at,
    )


def _with_filesystem_suite_status(
    suite: ManagedEvaluationBaselineSuite,
    *,
    updated_at: str | None = None,
) -> ManagedEvaluationBaselineSuite:
    if suite.error_message is not None:
        status = "failed"
    elif suite.manifest_path is not None and suite.manifest_path.is_file():
        status = "ready"
    else:
        status = "not_created"
    materialized_at = suite.materialized_at
    if status == "ready" and materialized_at is None:
        materialized_at = updated_at
    return ManagedEvaluationBaselineSuite(
        id=suite.id,
        preset_id=suite.preset_id,
        preset_version=suite.preset_version,
        status=status,
        suite_dir=suite.suite_dir,
        manifest_path=suite.manifest_path,
        error_message=suite.error_message,
        created_at=suite.created_at,
        updated_at=updated_at or suite.updated_at,
        materialized_at=materialized_at,
    )
