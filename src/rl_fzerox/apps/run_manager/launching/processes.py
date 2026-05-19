# src/rl_fzerox/apps/run_manager/launching/processes.py
from __future__ import annotations

import subprocess
import threading


def reap_child_when_done(process: subprocess.Popen[bytes]) -> None:
    """Wait for a manager child in the background so it cannot stay defunct."""

    thread = threading.Thread(
        target=process.wait,
        name=f"run-manager-reap-{process.pid}",
        daemon=True,
    )
    thread.start()
