# src/rl_fzerox/core/training/runs/migration.py
from __future__ import annotations

import shutil
from pathlib import Path

from rl_fzerox.core.training.runs.paths import RUN_LAYOUT

_POLICY_METADATA_SUFFIX = ".metadata.json"


def migrate_run_artifact_layout(run_dir: Path) -> None:
    """Move one legacy root-artifact run into the canonical checkpoint tree."""

    resolved_run_dir = run_dir.expanduser().resolve()
    for old_path, new_path in _legacy_to_canonical_artifact_paths(resolved_run_dir):
        if not old_path.exists():
            continue
        if new_path.exists():
            raise FileExistsError(
                "Run artifact migration would overwrite an existing canonical path: "
                f"{new_path}"
            )
        new_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(old_path), str(new_path))


def _legacy_to_canonical_artifact_paths(run_dir: Path) -> tuple[tuple[Path, Path], ...]:
    latest_policy = run_dir / RUN_LAYOUT.policy_artifacts.latest
    best_policy = run_dir / RUN_LAYOUT.policy_artifacts.best
    final_policy = run_dir / RUN_LAYOUT.policy_artifacts.final
    return (
        (run_dir / "latest_model.zip", run_dir / RUN_LAYOUT.model_artifacts.latest),
        (run_dir / "best_model.zip", run_dir / RUN_LAYOUT.model_artifacts.best),
        (run_dir / "final_model.zip", run_dir / RUN_LAYOUT.model_artifacts.final),
        (run_dir / "latest_policy.zip", latest_policy),
        (run_dir / "best_policy.zip", best_policy),
        (run_dir / "final_policy.zip", final_policy),
        (run_dir / "latest_policy.metadata.json", _policy_metadata_path(latest_policy)),
        (run_dir / "best_policy.metadata.json", _policy_metadata_path(best_policy)),
        (run_dir / "final_policy.metadata.json", _policy_metadata_path(final_policy)),
    )


def _policy_metadata_path(policy_path: Path) -> Path:
    return policy_path.with_name(f"{policy_path.stem}{_POLICY_METADATA_SUFFIX}")
