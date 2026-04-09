# src/rl_fzerox/core/training/runs/artifacts.py
from __future__ import annotations

from pathlib import Path


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
        latest_filename="latest_model.zip",
        best_filename="best_model.zip",
        final_filename="final_model.zip",
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
        latest_filename="latest_policy.zip",
        best_filename="best_policy.zip",
        final_filename="final_policy.zip",
    )


def _resolve_artifact_path(
    *,
    run_dir: Path,
    artifact: str,
    latest_filename: str,
    best_filename: str,
    final_filename: str,
) -> Path:
    resolved_run_dir = run_dir.expanduser().resolve()
    if artifact == "latest":
        preferred_filenames = (latest_filename, final_filename, best_filename)
    elif artifact == "best":
        preferred_filenames = (best_filename,)
    elif artifact == "final":
        preferred_filenames = (final_filename,)
    else:
        raise ValueError(f"Unsupported artifact kind: {artifact!r}")

    for filename in preferred_filenames:
        resolved_path = resolved_run_dir / filename
        if resolved_path.is_file():
            return resolved_path

    raise FileNotFoundError(f"No artifact could be found under run directory {resolved_run_dir}")
