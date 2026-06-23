# src/rl_fzerox/core/manager/transfer/files.py
"""Run-directory file selection and bundle path mapping."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path, PurePosixPath

from rl_fzerox.core.manager.transfer.models import RunBundleLayout


def run_files(run_dir: Path) -> Iterable[Path]:
    """Yield regular run files that are safe to include in a bundle payload."""

    for file_path in sorted(run_dir.rglob("*")):
        if file_path.is_symlink() or not file_path.is_file():
            continue
        yield file_path


def archive_path_for_run_file(
    layout: RunBundleLayout,
    file_path: Path,
    *,
    run_dir: Path,
) -> str:
    """Return the POSIX archive payload path for one run-local file."""

    return str(PurePosixPath(layout.payload_dir, file_path.relative_to(run_dir).as_posix()))
