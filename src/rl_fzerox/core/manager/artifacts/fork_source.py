# src/rl_fzerox/core/manager/artifacts/fork_source.py
"""Pinned checkpoint snapshots used for managed forks and resumes.

Each pinned fork source must be self-contained enough for a resumed child run
to validate and reload its source checkpoint without reaching back into the
parent run directory. That includes the resolved checkpoint artifacts and the
saved train manifest used by resume/preload safety checks.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Literal

from rl_fzerox.core.training.runs import (
    RUN_LAYOUT,
    resolve_model_artifact_path,
    resolve_policy_artifact_path,
    resolve_train_run_config_path,
)
from rl_fzerox.core.training.session import load_policy_artifact_metadata

ForkArtifact = Literal["latest", "best"]


def draft_fork_source_dir(*, manager_db_path: Path, draft_id: str) -> Path:
    """Return the pinned fork-source directory for one persisted draft."""

    return manager_db_path.expanduser().resolve().parent / "fork_sources" / "drafts" / draft_id


def run_fork_source_dir(*, run_dir: Path) -> Path:
    """Return the pinned fork-source directory owned by one child run."""

    return run_dir.expanduser().resolve() / "fork_source"


def snapshot_fork_source(
    *,
    source_run_dir: Path,
    artifact: ForkArtifact,
    destination_dir: Path,
) -> int:
    """Copy one exact parent checkpoint into a stable fork-source directory."""

    resolved_source_run_dir = source_run_dir.expanduser().resolve()
    resolved_destination_dir = destination_dir.expanduser().resolve()
    model_path = resolve_model_artifact_path(resolved_source_run_dir, artifact=artifact)
    policy_path = resolve_policy_artifact_path(resolved_source_run_dir, artifact=artifact)
    metadata = load_policy_artifact_metadata(policy_path)
    if metadata is None or metadata.num_timesteps is None:
        raise ValueError(
            "Could not determine checkpoint step for fork source "
            f"{artifact} in {resolved_source_run_dir}"
        )

    reset_fork_source_dir(resolved_destination_dir)
    resolved_destination_dir.mkdir(parents=True, exist_ok=True)
    source_config_path = resolve_train_run_config_path(resolved_source_run_dir)
    destination_config_path = resolved_destination_dir / RUN_LAYOUT.config_filename
    _copy_checkpoint_artifact_dir(
        source_dirs=(model_path.parent, policy_path.parent),
        destination_dir=resolved_destination_dir / RUN_LAYOUT.checkpoints_dirname / artifact,
    )
    link_or_copy_file(source_config_path, destination_config_path)
    return metadata.num_timesteps


def clone_fork_source(*, source_dir: Path, destination_dir: Path) -> None:
    """Clone one already-pinned fork source into a child run directory."""

    resolved_source_dir = source_dir.expanduser().resolve()
    resolved_destination_dir = destination_dir.expanduser().resolve()
    if not resolved_source_dir.is_dir():
        raise FileNotFoundError(f"Pinned fork source directory not found: {resolved_source_dir}")
    reset_fork_source_dir(resolved_destination_dir)
    shutil.copytree(
        resolved_source_dir,
        resolved_destination_dir,
        copy_function=_link_or_copy_file_str,
    )


def is_complete_fork_source(*, source_dir: Path, artifact: ForkArtifact) -> bool:
    """Return whether one pinned fork source is complete enough for warm start."""

    resolved_source_dir = source_dir.expanduser().resolve()
    if not resolved_source_dir.is_dir():
        return False
    try:
        resolve_model_artifact_path(resolved_source_dir, artifact=artifact)
        policy_path = resolve_policy_artifact_path(resolved_source_dir, artifact=artifact)
        resolve_train_run_config_path(resolved_source_dir)
    except FileNotFoundError:
        return False
    metadata = load_policy_artifact_metadata(policy_path)
    if metadata is None or metadata.num_timesteps is None:
        return False
    return True


def reset_fork_source_dir(path: Path) -> None:
    """Remove one pinned fork-source directory if it exists."""

    resolved_path = path.expanduser().resolve()
    if resolved_path.exists():
        shutil.rmtree(resolved_path)


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
                link_or_copy_file(source_path, destination_dir / source_path.name)


def link_or_copy_file(source: Path, destination: Path) -> Path:
    """Hardlink one file when possible, falling back to a metadata-preserving copy."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    try:
        destination.hardlink_to(source)
    except OSError:
        shutil.copy2(source, destination)
    return destination


def _link_or_copy_file_str(source: str, destination: str) -> object:
    return link_or_copy_file(Path(source), Path(destination))
