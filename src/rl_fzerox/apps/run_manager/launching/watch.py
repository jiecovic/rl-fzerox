# src/rl_fzerox/apps/run_manager/launching/watch.py
from __future__ import annotations

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
WatchRenderer = Literal["angrylion", "gliden64"]


def launch_watch_artifact(
    *,
    store: ManagerStore,
    run_id: str,
    artifact: str,
    device: Literal["cpu", "cuda"],
    renderer: WatchRenderer | None,
) -> WatchLaunchStatus:
    run = store.get_run(run_id)
    if run is None:
        raise ValueError(f"run not found: {run_id}")
    if artifact not in {"latest", "best"}:
        raise ValueError(f"unsupported watch artifact: {artifact}")
    resolve_model_artifact_path(run.run_dir, artifact=artifact)
    lease_id = store.viewer_lease_id(
        kind="run_watch",
        owner_id=run.id,
        qualifier=artifact,
    )
    if (
        active_watch_pid(
            store=store,
            lease_id=lease_id,
            run_id=run.id,
            run_dir=run.run_dir,
            artifact=artifact,
        )
        is not None
    ):
        return "already_running"
    overrides = watch_config_overrides(device=device, renderer=renderer)
    resolve_watch_app_config(
        policy_run_dir=None,
        policy_artifact="best" if artifact == "best" else "latest",
        manager_db_path=store.db_path,
        managed_run_id=run.id,
        session_name=lease_id,
        overrides=overrides,
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
        "--viewer-lease-id",
        lease_id,
        "--",
        *overrides,
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
    store.upsert_viewer_lease(
        lease_id=lease_id,
        kind="run_watch",
        owner_id=run.id,
        pid=process.pid,
        qualifier=artifact,
    )
    try:
        raise_if_watch_exited_early(process=process, log_path=log_path)
    except RuntimeError:
        store.clear_viewer_lease(lease_id=lease_id, pid=process.pid)
        raise
    reap_child_when_done(process)
    return "started"


def watch_config_overrides(
    *,
    device: Literal["cpu", "cuda"],
    renderer: WatchRenderer | None,
) -> tuple[str, ...]:
    overrides = [f"watch.device={device}"]
    if renderer is not None:
        overrides.append(f"emulator.renderer={renderer}")
    return tuple(overrides)


def manager_watch_log_path(run_id: str, *, artifact: str) -> Path:
    return (
        project_root_dir() / "local" / "manager" / "logs" / f"{run_id}.watch-{artifact}.log"
    ).resolve()


def raise_if_watch_exited_early(
    *,
    process: subprocess.Popen[bytes],
    log_path: Path,
) -> None:
    try:
        return_code = process.wait(timeout=0.35)
    except subprocess.TimeoutExpired:
        return

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


def active_watch_pid(
    *,
    store: ManagerStore,
    lease_id: str,
    run_id: str,
    run_dir: Path,
    artifact: str,
) -> int | None:
    lease = store.get_viewer_lease(lease_id)
    if lease is None:
        return None
    if lease.kind != "run_watch" or lease.owner_id != run_id or lease.qualifier != artifact:
        store.clear_viewer_lease(lease_id=lease_id)
        return None
    if watch_process_matches(
        pid=lease.pid,
        run_id=run_id,
        run_dir=run_dir,
        artifact=artifact,
    ):
        return lease.pid
    store.clear_viewer_lease(lease_id=lease_id, pid=lease.pid)
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
