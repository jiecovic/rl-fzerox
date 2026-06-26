# src/rl_fzerox/core/manager/artifacts/fork_source.py
"""Pinned checkpoint snapshots used for managed forks and resumes.

Each pinned fork source must be self-contained enough for a resumed child run
to validate and reload its source checkpoint without reaching back into a
mutable parent run directory. Local-run forks still resolve preload metadata
from SQLite. Snapshot-only forks additionally write a small loader metadata
file beside the copied checkpoint artifacts; the train manifest remains only a
human/export mirror and is never loaded as config.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from rl_fzerox.core.domain.policy import TrainAlgorithmName
from rl_fzerox.core.training.runs import (
    RUN_LAYOUT,
    resolve_model_artifact_path,
    resolve_policy_artifact_path,
    resolve_train_run_config_path,
)
from rl_fzerox.core.training.session import load_policy_artifact_metadata

ForkArtifact = Literal["latest", "best"]
FORK_SOURCE_METADATA_FILENAME = "fork_source.metadata.json"


@dataclass(frozen=True, slots=True)
class ForkSourceMetadata:
    """Checkpoint structure metadata needed by weights-only preload."""

    source_algorithm: TrainAlgorithmName
    source_auxiliary_state_enabled: bool
    source_auxiliary_state_head_arch: tuple[int, ...]


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


def snapshot_fork_source_from_paths(
    *,
    policy_path: Path,
    model_path: Path,
    artifact: ForkArtifact,
    destination_dir: Path,
    source_num_timesteps: int | None = None,
    engine_tuning_state_path: Path | None = None,
    engine_tuning_model_path: Path | None = None,
) -> int:
    """Copy explicit checkpoint files into the canonical fork-source layout."""

    resolved_policy_path = policy_path.expanduser().resolve()
    metadata = load_policy_artifact_metadata(resolved_policy_path)
    if source_num_timesteps is None:
        if metadata is None or metadata.num_timesteps is None:
            raise ValueError(f"Could not determine checkpoint step for {resolved_policy_path}")
        source_num_timesteps = metadata.num_timesteps

    resolved_destination_dir = destination_dir.expanduser().resolve()
    reset_fork_source_dir(resolved_destination_dir)
    checkpoint_dir = resolved_destination_dir / RUN_LAYOUT.checkpoints_dirname / artifact
    link_or_copy_file(model_path.expanduser().resolve(), checkpoint_dir / "model.zip")
    link_or_copy_file(resolved_policy_path, checkpoint_dir / "policy.zip")
    metadata_path = resolved_policy_path.with_name(f"{resolved_policy_path.stem}.metadata.json")
    if metadata_path.is_file():
        link_or_copy_file(metadata_path, checkpoint_dir / metadata_path.name)
    if engine_tuning_state_path is not None:
        link_or_copy_file(
            engine_tuning_state_path.expanduser().resolve(),
            checkpoint_dir / RUN_LAYOUT.engine_tuning_state_filename,
        )
    if engine_tuning_model_path is not None:
        link_or_copy_file(
            engine_tuning_model_path.expanduser().resolve(),
            checkpoint_dir / RUN_LAYOUT.engine_tuning_model_filename,
        )
    return source_num_timesteps


def write_fork_source_metadata(
    *,
    source_dir: Path,
    metadata: ForkSourceMetadata,
) -> Path:
    """Write source-checkpoint loader metadata for one snapshot-only fork."""

    metadata_path = fork_source_metadata_path(source_dir)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_algorithm": metadata.source_algorithm,
                "source_auxiliary_state_enabled": metadata.source_auxiliary_state_enabled,
                "source_auxiliary_state_head_arch": list(
                    metadata.source_auxiliary_state_head_arch
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return metadata_path


def load_fork_source_metadata(*, source_dir: Path) -> ForkSourceMetadata:
    """Read source-checkpoint loader metadata from one snapshot-only fork."""

    metadata_path = fork_source_metadata_path(source_dir)
    raw_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(raw_metadata, dict):
        raise ValueError(f"Fork source metadata must be a JSON object: {metadata_path}")
    if raw_metadata.get("schema_version") != 1:
        raise ValueError(f"Unsupported fork source metadata schema: {metadata_path}")
    return ForkSourceMetadata(
        source_algorithm=_metadata_algorithm(raw_metadata, metadata_path=metadata_path),
        source_auxiliary_state_enabled=_metadata_bool(
            raw_metadata,
            "source_auxiliary_state_enabled",
            metadata_path=metadata_path,
        ),
        source_auxiliary_state_head_arch=_metadata_int_tuple(
            raw_metadata,
            "source_auxiliary_state_head_arch",
            metadata_path=metadata_path,
        ),
    )


def fork_source_metadata_path(source_dir: Path) -> Path:
    return source_dir.expanduser().resolve() / FORK_SOURCE_METADATA_FILENAME


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


def _metadata_algorithm(
    metadata: dict[object, object],
    *,
    metadata_path: Path,
) -> TrainAlgorithmName:
    value = metadata.get("source_algorithm")
    if value == "maskable_hybrid_action_ppo":
        return "maskable_hybrid_action_ppo"
    if value == "maskable_hybrid_recurrent_ppo":
        return "maskable_hybrid_recurrent_ppo"
    raise ValueError(f"Fork source metadata has invalid source_algorithm: {metadata_path}")


def _metadata_bool(
    metadata: dict[object, object],
    key: str,
    *,
    metadata_path: Path,
) -> bool:
    value = metadata.get(key)
    if isinstance(value, bool):
        return value
    raise ValueError(f"Fork source metadata has invalid {key}: {metadata_path}")


def _metadata_int_tuple(
    metadata: dict[object, object],
    key: str,
    *,
    metadata_path: Path,
) -> tuple[int, ...]:
    values = metadata.get(key)
    if not isinstance(values, list):
        raise ValueError(f"Fork source metadata has invalid {key}: {metadata_path}")
    result: list[int] = []
    for value in values:
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"Fork source metadata has invalid {key}: {metadata_path}")
        result.append(value)
    return tuple(result)


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
