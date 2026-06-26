# src/rl_fzerox/core/manager/policy_sources.py
"""Resolve assignable policy checkpoint sources for manager-owned workflows.

Save-game course setup can point at a moving run artifact or at an immutable
evaluation snapshot. This module owns that distinction so runner code receives
one resolved policy source instead of guessing from run ids or filesystem paths.
"""

from __future__ import annotations

from pathlib import Path

from sqlalchemy.orm import Session

from rl_fzerox.core.manager.db.repositories import evaluations as evaluation_repository
from rl_fzerox.core.manager.db.repositories import runs as run_repository
from rl_fzerox.core.manager.models import (
    ManagedPolicySource,
    ManagedRun,
    PolicySourceArtifact,
    PolicySourceKind,
)
from rl_fzerox.core.training.runs import resolve_policy_artifact_path


def resolve_policy_source(
    session: Session,
    *,
    kind: PolicySourceKind,
    source_id: str,
    artifact: PolicySourceArtifact,
    require_policy_artifact: bool = False,
) -> ManagedPolicySource:
    """Resolve one save-game policy source from SQLite-owned manager state."""

    match kind:
        case "run":
            return _run_policy_source(
                session,
                run_id=source_id,
                artifact=artifact,
                require_policy_artifact=require_policy_artifact,
            )
        case "evaluation":
            return _evaluation_policy_source(
                session,
                evaluation_id=source_id,
                artifact=artifact,
                require_policy_artifact=require_policy_artifact,
            )


def _run_policy_source(
    session: Session,
    *,
    run_id: str,
    artifact: PolicySourceArtifact,
    require_policy_artifact: bool,
) -> ManagedPolicySource:
    run = run_repository.get_managed_run(session, run_id)
    if run is None:
        raise KeyError("policy run not found")
    policy_path = _run_policy_path(
        run_dir=run.run_dir,
        artifact=artifact,
        require_policy_artifact=require_policy_artifact,
    )
    return ManagedPolicySource(
        kind="run",
        id=run.id,
        name=run.name,
        artifact=artifact,
        config=run.config,
        source_dir=run.run_dir,
        mutable=True,
        created_at=run.created_at,
        updated_at=run.worker_heartbeat_at or run.stopped_at or run.started_at or run.created_at,
        policy_path=policy_path,
        source_run_id=run.id,
        source_run_name=run.name,
        lineage_num_timesteps=_run_lineage_timesteps(run),
    )


def _evaluation_policy_source(
    session: Session,
    *,
    evaluation_id: str,
    artifact: PolicySourceArtifact,
    require_policy_artifact: bool,
) -> ManagedPolicySource:
    evaluation = evaluation_repository.get_managed_evaluation(session, evaluation_id)
    if evaluation is None:
        raise KeyError("policy evaluation not found")
    if evaluation.status != "completed":
        raise ValueError("policy evaluation is not completed")
    if evaluation.checkpoint.artifact != artifact:
        raise ValueError(
            f"policy evaluation snapshot uses {evaluation.checkpoint.artifact!r}, not {artifact!r}"
        )
    policy_path = Path(evaluation.checkpoint.copied_policy_path).expanduser().resolve()
    if require_policy_artifact and not policy_path.is_file():
        raise FileNotFoundError(f"evaluation policy checkpoint is missing: {policy_path}")
    return ManagedPolicySource(
        kind="evaluation",
        id=evaluation.id,
        name=evaluation.name,
        artifact=artifact,
        config=evaluation.config,
        source_dir=_checkpoint_snapshot_root(policy_path),
        mutable=False,
        created_at=evaluation.created_at,
        updated_at=evaluation.updated_at,
        policy_path=policy_path,
        source_run_id=evaluation.source_run_id or evaluation.checkpoint.source_run_id,
        source_run_name=evaluation.checkpoint.source_run_name,
        local_num_timesteps=evaluation.checkpoint.local_num_timesteps,
        lineage_num_timesteps=evaluation.checkpoint.lineage_num_timesteps,
    )


def _run_policy_path(
    *,
    run_dir: Path,
    artifact: PolicySourceArtifact,
    require_policy_artifact: bool,
) -> Path | None:
    if not require_policy_artifact:
        return None
    return resolve_policy_artifact_path(run_dir, artifact=artifact)


def _run_lineage_timesteps(run: ManagedRun) -> int | None:
    runtime = run.runtime
    if runtime is None:
        return None
    return run.lineage_step_offset + runtime.num_timesteps


def _checkpoint_snapshot_root(policy_path: Path) -> Path:
    if policy_path.name != "policy.zip" or policy_path.parent.parent.name != "checkpoints":
        raise ValueError(f"unsupported evaluation policy snapshot path: {policy_path}")
    return policy_path.parent.parent.parent
