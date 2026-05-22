# src/rl_fzerox/apps/run_manager/launching/watch.py
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Literal

from rl_fzerox.apps.run_manager.launching.processes import reap_child_when_done
from rl_fzerox.apps.watch_cli.resolve import resolve_watch_app_config
from rl_fzerox.core.manager import ManagerStore
from rl_fzerox.core.runtime_spec.paths import project_root_dir
from rl_fzerox.core.training.runs import resolve_model_artifact_path

WatchLaunchStatus = Literal["started", "already_running"]


def launch_watch_artifact(
    *,
    store: ManagerStore,
    run_id: str,
    artifact: str,
    device: Literal["cpu", "cuda"],
) -> WatchLaunchStatus:
    run = store.get_run(run_id)
    if run is None:
        raise ValueError(f"run not found: {run_id}")
    if artifact not in {"latest", "best"}:
        raise ValueError(f"unsupported watch artifact: {artifact}")
    resolve_model_artifact_path(run.run_dir, artifact=artifact)
    pid_path = manager_watch_pid_path(run.id, artifact=artifact)
    if (
        active_watch_pid(
            pid_path=pid_path,
            run_id=run.id,
            run_dir=run.run_dir,
            artifact=artifact,
        )
        is not None
    ):
        return "already_running"
    resolve_watch_app_config(
        policy_run_dir=None,
        policy_artifact="best" if artifact == "best" else "latest",
        manager_db_path=store.db_path,
        managed_run_id=run.id,
        overrides=(f"watch.device={device}",),
    )
    log_path = manager_watch_log_path(run.id, artifact=artifact)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "rl_fzerox.apps.watch",
        "--manager-db-path",
        str(store.db_path),
        "--managed-run-id",
        run.id,
        "--artifact",
        artifact,
        "--watch-pid-file",
        str(pid_path),
        "--",
        f"watch.device={device}",
    ]
    with log_path.open("ab") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=project_root_dir(),
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    write_watch_pid_file(
        pid_path=pid_path,
        pid=process.pid,
        run_id=run.id,
        run_dir=run.run_dir,
        artifact=artifact,
    )
    raise_if_watch_exited_early(process=process, log_path=log_path, pid_path=pid_path)
    reap_child_when_done(process)
    return "started"


def manager_watch_log_path(run_id: str, *, artifact: str) -> Path:
    return (
        project_root_dir() / "local" / "manager" / "logs" / f"{run_id}.watch-{artifact}.log"
    ).resolve()


def manager_watch_pid_path(run_id: str, *, artifact: str) -> Path:
    return (
        project_root_dir() / "local" / "manager" / "watch" / f"{run_id}.watch-{artifact}.json"
    ).resolve()


def raise_if_watch_exited_early(
    *,
    process: subprocess.Popen[bytes],
    log_path: Path,
    pid_path: Path,
) -> None:
    try:
        return_code = process.wait(timeout=0.35)
    except subprocess.TimeoutExpired:
        return

    pid_path.unlink(missing_ok=True)
    detail = watch_failure_detail(log_path)
    if detail is None:
        raise RuntimeError(f"watch exited immediately with code {return_code}; see {log_path}")
    raise RuntimeError(f"watch exited immediately with code {return_code}: {detail}")


def watch_failure_detail(log_path: Path) -> str | None:
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        detail = line.strip()
        if detail:
            return detail
    return None


def active_watch_pid(*, pid_path: Path, run_id: str, run_dir: Path, artifact: str) -> int | None:
    payload = read_watch_pid_file(pid_path)
    if payload is None:
        return None
    pid = payload.get("pid")
    if not isinstance(pid, int):
        pid_path.unlink(missing_ok=True)
        return None
    if watch_process_matches(pid=pid, run_id=run_id, run_dir=run_dir, artifact=artifact):
        return pid
    pid_path.unlink(missing_ok=True)
    return None


def write_watch_pid_file(
    *,
    pid_path: Path,
    pid: int,
    run_id: str,
    run_dir: Path,
    artifact: str,
) -> None:
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(
        json.dumps(
            {
                "pid": pid,
                "run_id": run_id,
                "run_dir": str(run_dir),
                "artifact": artifact,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def read_watch_pid_file(pid_path: Path) -> dict[str, object] | None:
    try:
        return json.loads(pid_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError):
        pid_path.unlink(missing_ok=True)
        return None


def watch_process_matches(*, pid: int, run_id: str, run_dir: Path, artifact: str) -> bool:
    proc_dir = Path("/proc") / str(pid)
    if not proc_dir.is_dir():
        return False
    try:
        cmdline = (proc_dir / "cmdline").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    normalized = cmdline.replace("\x00", " ")
    return (
        "rl_fzerox.apps.watch" in normalized
        and f"--artifact {artifact}" in normalized
        and (f"--managed-run-id {run_id}" in normalized or f"--run-dir {run_dir}" in normalized)
    )
