# src/rl_fzerox/core/manager/artifacts/paths.py
"""Managed-run directory conventions under the local artifact tree."""

from __future__ import annotations

from pathlib import Path

_DEFAULT_MANAGER_RUNS_ROOT = Path("local/runs").resolve()
_DEFAULT_TENSORBOARD_VIEWS_ROOT = Path("local/tensorboard_views").resolve()


def manager_runs_root(*, output_root: Path | None = None) -> Path:
    """Return the manager-owned run root under the normal training output tree."""

    return (output_root or _DEFAULT_MANAGER_RUNS_ROOT).resolve()


def manager_tensorboard_views_root(*, output_root: Path | None = None) -> Path:
    """Return the symlink-only grouped TensorBoard view root."""

    if output_root is None:
        return _DEFAULT_TENSORBOARD_VIEWS_ROOT
    return output_root.expanduser().resolve().parent / "tensorboard_views"


def predicted_managed_lineage_dir(
    lineage_id: str,
    *,
    output_root: Path | None = None,
) -> Path:
    """Return the exact lineage directory for one manager lineage id."""

    return manager_runs_root(output_root=output_root) / lineage_id


def predicted_managed_run_dir(
    run_id: str,
    *,
    lineage_id: str,
    output_root: Path | None = None,
) -> Path:
    """Return the exact manager-owned run directory for one run inside one lineage."""

    return predicted_managed_lineage_dir(lineage_id, output_root=output_root) / run_id
