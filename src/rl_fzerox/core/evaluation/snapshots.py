# src/rl_fzerox/core/evaluation/snapshots.py
"""Copy immutable checkpoint snapshots for evaluation runs.

Evaluation must not read a moving run checkpoint while training continues. This
module copies or hardlinks the selected policy/model artifact into the
evaluation directory and records enough metadata to reproduce what was run.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from rl_fzerox.core.evaluation.models import (
    EvaluationCheckpointArtifact,
    EvaluationCheckpointSnapshot,
)
from rl_fzerox.core.training.runs import (
    RUN_LAYOUT,
    resolve_model_artifact_path,
    resolve_policy_artifact_path,
)
from rl_fzerox.core.training.session import load_policy_artifact_metadata


@dataclass(frozen=True, slots=True)
class EvaluationCheckpointSource:
    """Managed run artifact selected as the source for an evaluation snapshot."""

    run_id: str | None
    run_name: str | None
    run_dir: Path
    artifact: EvaluationCheckpointArtifact
    lineage_step_offset: int = 0
    policy_path: Path | None = None
    model_path: Path | None = None
    engine_tuning_state_path: Path | None = None
    engine_tuning_model_path: Path | None = None
    local_num_timesteps: int | None = None
    lineage_num_timesteps: int | None = None


def snapshot_evaluation_checkpoint(
    source: EvaluationCheckpointSource,
    *,
    destination_dir: Path,
) -> EvaluationCheckpointSnapshot:
    """Copy one exact policy artifact into an immutable evaluation directory.

    The destination uses the normal run checkpoint layout under
    ``checkpoints/<artifact>/`` so later policy resolvers can treat an
    evaluation snapshot similarly to a run directory. Unlike fork-source
    snapshots, this does not copy or depend on a train manifest.
    """

    resolved_source_run_dir = source.run_dir.expanduser().resolve()
    resolved_destination_dir = destination_dir.expanduser().resolve()
    _assert_empty_destination(resolved_destination_dir)

    model_path = _source_model_path(source, resolved_source_run_dir)
    policy_path = _source_policy_path(source, resolved_source_run_dir)
    metadata = load_policy_artifact_metadata(policy_path)
    if metadata is None or metadata.num_timesteps is None:
        raise ValueError(
            "Could not determine checkpoint step for evaluation source "
            f"{source.artifact} in {resolved_source_run_dir}"
        )

    tmp_dir = resolved_destination_dir.with_name(f".{resolved_destination_dir.name}.tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    try:
        checkpoint_dir = tmp_dir / RUN_LAYOUT.checkpoints_dirname / source.artifact
        _copy_checkpoint_artifact_dir(
            source_dirs=(model_path.parent, policy_path.parent),
            destination_dir=checkpoint_dir,
        )
        _copy_optional_sidecar(
            source.engine_tuning_state_path,
            checkpoint_dir / RUN_LAYOUT.engine_tuning_state_filename,
        )
        _copy_optional_sidecar(
            source.engine_tuning_model_path,
            checkpoint_dir / RUN_LAYOUT.engine_tuning_model_filename,
        )
        os.replace(tmp_dir, resolved_destination_dir)
    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)

    copied_checkpoint_dir = (
        resolved_destination_dir / RUN_LAYOUT.checkpoints_dirname / source.artifact
    )
    local_num_timesteps = (
        source.local_num_timesteps
        if source.local_num_timesteps is not None
        else metadata.num_timesteps
    )
    lineage_num_timesteps = (
        source.lineage_num_timesteps
        if source.lineage_num_timesteps is not None
        else metadata.lineage_num_timesteps
    )
    if lineage_num_timesteps is None and source.lineage_step_offset > 0:
        lineage_num_timesteps = source.lineage_step_offset + local_num_timesteps
    return EvaluationCheckpointSnapshot(
        source_run_id=source.run_id,
        source_run_name=source.run_name,
        artifact=source.artifact,
        source_policy_path=str(policy_path),
        copied_policy_path=str(copied_checkpoint_dir / policy_path.name),
        source_model_path=str(model_path),
        copied_model_path=str(copied_checkpoint_dir / model_path.name),
        local_num_timesteps=local_num_timesteps,
        lineage_num_timesteps=lineage_num_timesteps,
        source_mtime_ns=policy_path.stat().st_mtime_ns,
    )


def _source_model_path(source: EvaluationCheckpointSource, resolved_source_run_dir: Path) -> Path:
    if source.model_path is not None:
        return source.model_path.expanduser().resolve()
    return resolve_model_artifact_path(resolved_source_run_dir, artifact=source.artifact)


def _source_policy_path(source: EvaluationCheckpointSource, resolved_source_run_dir: Path) -> Path:
    if source.policy_path is not None:
        return source.policy_path.expanduser().resolve()
    return resolve_policy_artifact_path(resolved_source_run_dir, artifact=source.artifact)


def _assert_empty_destination(destination_dir: Path) -> None:
    if not destination_dir.exists():
        return
    if any(destination_dir.iterdir()):
        raise FileExistsError(
            "Evaluation checkpoint snapshot destination already exists and is not empty: "
            f"{destination_dir}"
        )
    destination_dir.rmdir()


def _copy_checkpoint_artifact_dir(
    *,
    source_dirs: tuple[Path, ...],
    destination_dir: Path,
) -> None:
    copied_sources: set[Path] = set()
    destination_dir.mkdir(parents=True, exist_ok=True)
    for source_dir in source_dirs:
        resolved_source_dir = source_dir.expanduser().resolve()
        if resolved_source_dir in copied_sources:
            continue
        copied_sources.add(resolved_source_dir)
        for source_path in sorted(resolved_source_dir.iterdir()):
            if source_path.is_file():
                _link_or_copy_file(source_path, destination_dir / source_path.name)


def _copy_optional_sidecar(source: Path | None, destination: Path) -> None:
    if source is not None:
        _link_or_copy_file(source.expanduser().resolve(), destination)


def _link_or_copy_file(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    try:
        destination.hardlink_to(source)
    except OSError:
        shutil.copy2(source, destination)
    return destination
