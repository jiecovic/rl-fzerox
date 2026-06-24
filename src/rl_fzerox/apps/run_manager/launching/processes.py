# src/rl_fzerox/apps/run_manager/launching/processes.py
from __future__ import annotations

import shlex
import subprocess
import threading
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO


@contextmanager
def fresh_process_log(
    log_path: Path,
    *,
    command: Sequence[str],
    cwd: Path,
) -> Generator[BinaryIO]:
    """Open a per-launch process log without carrying stale failures forward."""

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("wb") as log_handle:
        log_handle.write(_launch_log_header(command=command, cwd=cwd).encode("utf-8"))
        log_handle.flush()
        yield log_handle


def reap_child_when_done(process: subprocess.Popen[bytes]) -> None:
    """Wait for a manager child in the background so it cannot stay defunct."""

    thread = threading.Thread(
        target=process.wait,
        name=f"run-manager-reap-{process.pid}",
        daemon=True,
    )
    thread.start()


def _launch_log_header(*, command: Sequence[str], cwd: Path) -> str:
    launched_at = datetime.now(UTC).isoformat(timespec="seconds")
    return f"# launched_at={launched_at}\n# cwd={cwd}\n# command={shlex.join(tuple(command))}\n\n"
