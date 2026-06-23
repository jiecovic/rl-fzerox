# src/rl_fzerox/apps/run_manager/launching/evaluations.py
"""Launch detached evaluation workers from the run manager."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Literal

from rl_fzerox.apps.run_manager.launching.processes import reap_child_when_done
from rl_fzerox.core.manager import ManagedEvaluation, ManagerStore
from rl_fzerox.core.runtime_spec.paths import project_root_dir


def manager_evaluation_log_path(evaluation_id: str) -> Path:
    return (
        project_root_dir() / "local" / "manager" / "logs" / f"{evaluation_id}.evaluation.log"
    ).resolve()


def launch_evaluation_worker(
    store: ManagerStore,
    *,
    evaluation_id: str,
    device: Literal["cpu", "cuda"],
    worker_count: int = 1,
) -> ManagedEvaluation:
    """Start one new or failed evaluation and return its running DB row."""

    if worker_count < 1:
        raise ValueError(f"worker_count must be at least 1, got {worker_count}")
    evaluation = store.get_evaluation(evaluation_id)
    if evaluation is None:
        raise ValueError("evaluation not found")
    if evaluation.status not in {"created", "failed", "cancelled"}:
        raise ValueError(
            "only created, failed, or cancelled evaluations can be started, "
            f"got {evaluation.status}"
        )

    running = store.mark_evaluation_running(evaluation_id)
    log_path = manager_evaluation_log_path(evaluation_id)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "rl_fzerox.apps.evaluation_worker",
        "--db-path",
        str(store.db_path),
        "--evaluation-id",
        evaluation_id,
        "--device",
        device,
        "--worker-count",
        str(worker_count),
    ]
    try:
        with log_path.open("ab") as log_handle:
            process = subprocess.Popen(
                command,
                cwd=project_root_dir(),
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        reap_child_when_done(process)
    except Exception as exc:
        store.mark_evaluation_failed(evaluation_id, error_message=str(exc))
        raise
    return running
