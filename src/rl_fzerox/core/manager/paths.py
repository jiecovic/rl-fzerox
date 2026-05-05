# src/rl_fzerox/core/manager/paths.py
from __future__ import annotations

from pathlib import Path

_DEFAULT_MANAGER_RUNS_ROOT = Path("local/runs").resolve()


def manager_runs_root(*, output_root: Path | None = None) -> Path:
    """Return the manager-owned run root under the normal training output tree."""

    return (output_root or _DEFAULT_MANAGER_RUNS_ROOT).resolve()


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
