# src/rl_fzerox/core/training/runs/artifacts.py
from __future__ import annotations

from pathlib import Path

from rl_fzerox.core.training.runs.paths import (
    MODEL_ARTIFACT_FILENAMES,
    POLICY_ARTIFACT_FILENAMES,
    ArtifactFilenames,
)


def resolve_latest_model_path(run_dir: Path) -> Path:
    """Resolve the newest full-model artifact from a run directory."""

    return resolve_model_artifact_path(run_dir, artifact="latest")


def resolve_latest_policy_path(run_dir: Path) -> Path:
    """Resolve the newest policy-only artifact from a run directory."""

    return resolve_policy_artifact_path(run_dir, artifact="latest")


def resolve_model_artifact_path(
    run_dir: Path,
    *,
    artifact: str,
) -> Path:
    """Resolve one full-model artifact from a run directory."""

    return _resolve_artifact_path(
        run_dir=run_dir,
        artifact=artifact,
        filenames=MODEL_ARTIFACT_FILENAMES,
    )


def resolve_policy_artifact_path(
    run_dir: Path,
    *,
    artifact: str,
) -> Path:
    """Resolve one policy-only artifact from a run directory."""

    return _resolve_artifact_path(
        run_dir=run_dir,
        artifact=artifact,
        filenames=POLICY_ARTIFACT_FILENAMES,
    )


def _resolve_artifact_path(
    *,
    run_dir: Path,
    artifact: str,
    filenames: ArtifactFilenames,
) -> Path:
    resolved_run_dir = run_dir.expanduser().resolve()
    if artifact == "latest":
        preferred_filenames = (filenames.latest, filenames.final, filenames.best)
    elif artifact == "best":
        preferred_filenames = (filenames.best,)
    elif artifact == "final":
        preferred_filenames = (filenames.final,)
    else:
        raise ValueError(f"Unsupported artifact kind: {artifact!r}")

    for filename in preferred_filenames:
        resolved_path = resolved_run_dir / filename
        if resolved_path.is_file():
            return resolved_path

    raise FileNotFoundError(f"No artifact could be found under run directory {resolved_run_dir}")
