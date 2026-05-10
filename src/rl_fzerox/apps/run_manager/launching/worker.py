# src/rl_fzerox/apps/run_manager/launching/worker.py
from __future__ import annotations

import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from rl_fzerox.core.manager import ManagerStore
from rl_fzerox.core.runtime_spec.paths import project_root_dir


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def manager_worker_log_path(run_id: str) -> Path:
    return (project_root_dir() / "local" / "manager" / "logs" / f"{run_id}.log").resolve()


def spawn_manager_worker(
    *,
    store: ManagerStore,
    run_id: str,
    resume: bool,
) -> None:
    log_path = manager_worker_log_path(run_id)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    launch_token = uuid4().hex
    command = [
        sys.executable,
        "-m",
        "rl_fzerox.apps.run_manager.worker",
        "--db-path",
        str(store.db_path),
        "--run-id",
        run_id,
        "--launch-token",
        launch_token,
    ]
    if resume:
        command.append("--resume")
    with log_path.open("ab") as log_handle:
        try:
            process = subprocess.Popen(
                command,
                cwd=project_root_dir(),
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            registered = store.register_run_worker(
                run_id=run_id,
                launch_token=launch_token,
                pid=process.pid,
                launched_at=utc_now(),
            )
            if not registered:
                raise RuntimeError(f"managed run disappeared before worker registration: {run_id}")
        except Exception:
            store.clear_run_worker(run_id)
            store.update_run_status(
                run_id=run_id,
                status="failed",
                stopped_at=utc_now(),
                message=f"failed to launch manager worker; see {log_path}",
            )
            raise
