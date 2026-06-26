# src/rl_fzerox/apps/run_manager/launching/watch.py
from __future__ import annotations

import os
import subprocess
import sys
import threading
from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from typing import Literal, Protocol

from rl_fzerox.apps.run_manager.launching.processes import fresh_process_log
from rl_fzerox.core.manager import ManagerStore
from rl_fzerox.core.manager.projection.watch import resolve_watch_app_config
from rl_fzerox.core.manager.registry.viewers import VIEWER_LEASE_POLICY, viewer_lease_is_fresh
from rl_fzerox.core.runtime_spec.paths import project_root_dir
from rl_fzerox.core.runtime_spec.renderers import RendererName
from rl_fzerox.core.training.runs import resolve_model_artifact_path

WatchLaunchStatus = Literal["started", "already_running"]
type WatchRenderer = RendererName
WATCH_STARTUP_TIMEOUT_SECONDS = 2.0
_WATCH_LAUNCH_LOCK = threading.Lock()


class _WatchProcess(Protocol):
    pid: int

    def wait(self, timeout: float | None = None) -> int: ...


def launch_watch_artifact(
    *,
    store: ManagerStore,
    run_id: str,
    artifact: str,
    device: Literal["cpu", "cuda"],
    renderer: WatchRenderer | None,
    deterministic_policy: bool,
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
    startup_lease = _WatchStartupLease(store=store, run_id=run.id, artifact=artifact)
    if (
        active_watch_pid(
            store=store,
            lease_id=lease_id,
            run_id=run.id,
            artifact=artifact,
        )
        is not None
        or startup_lease.is_active()
    ):
        return "already_running"
    if not startup_lease.acquire():
        return "already_running"
    startup_lease.start_heartbeat()
    try:
        validate_watch_device(device)
        overrides = watch_config_overrides(
            device=device,
            renderer=renderer,
            deterministic_policy=deterministic_policy,
        )
        resolve_watch_app_config(
            run_id=run.id,
            policy_artifact="best" if artifact == "best" else "latest",
            manager_db_path=store.db_path,
            session_name=lease_id,
            overrides=overrides,
            startup_reporter=watch_startup_reporter(store=store, run_id=run.id),
        )
        log_path = manager_watch_log_path(run.id, artifact=artifact)
        command = [
            sys.executable,
            "-m",
            "rl_fzerox.apps.watch",
            "--manager-db-path",
            str(store.db_path),
            "--run-id",
            run.id,
            "--artifact",
            artifact,
            "--viewer-lease-id",
            lease_id,
            "--",
            *overrides,
        ]
        cwd = project_root_dir()
        with fresh_process_log(log_path, command=command, cwd=cwd) as log_handle:
            process = subprocess.Popen(
                command,
                cwd=cwd,
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
        startup_lease.clear()
        try:
            raise_if_watch_exited_early(process=process, log_path=log_path)
        except RuntimeError as error:
            store.clear_viewer_lease(lease_id=lease_id, pid=process.pid)
            store.append_run_event(
                run_id=run.id,
                kind="watch_failed",
                message=f"{artifact} watch failed: {error}",
            )
            raise
        reap_watch_when_done(
            db_path=store.db_path,
            process=process,
            lease_id=lease_id,
            run_id=run.id,
            artifact=artifact,
            log_path=log_path,
        )
        return "started"
    finally:
        startup_lease.stop_heartbeat()
        startup_lease.clear()


class _WatchStartupLease:
    """Guard the materialization phase before a visible watch process exists."""

    def __init__(self, *, store: ManagerStore, run_id: str, artifact: str) -> None:
        self._store = store
        self._run_id = run_id
        self._artifact = artifact
        self._qualifier = f"{artifact}:launch"
        self._lease_id = store.viewer_lease_id(
            kind="run_watch",
            owner_id=run_id,
            qualifier=self._qualifier,
        )
        self._pid = os.getpid()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def is_active(self) -> bool:
        lease = self._store.get_viewer_lease(self._lease_id)
        if lease is None:
            return False
        if (
            lease.kind != "run_watch"
            or lease.owner_id != self._run_id
            or lease.qualifier != self._qualifier
        ):
            self._store.clear_viewer_lease(lease_id=self._lease_id)
            return False
        if not viewer_lease_is_fresh(lease):
            self._store.clear_viewer_lease(lease_id=self._lease_id, pid=lease.pid)
            return False
        return True

    def acquire(self) -> bool:
        with _WATCH_LAUNCH_LOCK:
            if self.is_active():
                return False
            self._store.upsert_viewer_lease(
                lease_id=self._lease_id,
                kind="run_watch",
                owner_id=self._run_id,
                pid=self._pid,
                qualifier=self._qualifier,
            )
        return True

    def start_heartbeat(self) -> None:
        self._thread = threading.Thread(
            target=self._heartbeat_loop,
            name=f"run-manager-watch-launch-{self._run_id}-{self._artifact}",
            daemon=True,
        )
        self._thread.start()

    def stop_heartbeat(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=VIEWER_LEASE_POLICY.heartbeat_interval.total_seconds() + 1.0)

    def clear(self) -> None:
        self._store.clear_viewer_lease(lease_id=self._lease_id, pid=self._pid)

    def _heartbeat_loop(self) -> None:
        interval = VIEWER_LEASE_POLICY.heartbeat_interval.total_seconds()
        while not self._stop_event.wait(interval):
            if not self._store.heartbeat_viewer_lease(
                lease_id=self._lease_id,
                pid=self._pid,
                heartbeat_at=self._store.utc_now(),
            ):
                return


def watch_config_overrides(
    *,
    device: Literal["cpu", "cuda"],
    renderer: WatchRenderer | None,
    deterministic_policy: bool,
) -> tuple[str, ...]:
    overrides = [
        f"watch.device={device}",
        f"watch.deterministic_policy={str(deterministic_policy).lower()}",
    ]
    if renderer is not None:
        overrides.append(f"emulator.renderer={renderer}")
    return tuple(overrides)


def validate_watch_device(device: Literal["cpu", "cuda"]) -> None:
    """Reject CUDA watch launches before spawning when this Python cannot use CUDA."""

    if device != "cuda":
        return
    try:
        torch = import_module("torch")
    except ImportError as exc:
        raise RuntimeError(
            "CUDA watch requested, but PyTorch is not installed. "
            "Select cpu or install a CUDA-enabled PyTorch build."
        ) from exc
    cuda = getattr(torch, "cuda", None)
    is_available = getattr(cuda, "is_available", None)
    if not callable(is_available) or not bool(is_available()):
        raise RuntimeError(
            "CUDA watch requested, but the active PyTorch build cannot use CUDA. "
            "Select cpu or install a CUDA-enabled PyTorch build."
        )


def watch_startup_reporter(*, store: ManagerStore, run_id: str) -> Callable[[str, str], None]:
    """Persist Watch-side baseline materialization progress in the run timeline."""

    def report(_kind: str, message: str) -> None:
        store.append_run_event(
            run_id=run_id,
            kind="startup_watch_materialize",
            message=message,
        )

    return report


def manager_watch_log_path(run_id: str, *, artifact: str) -> Path:
    return (
        project_root_dir() / "local" / "manager" / "logs" / f"{run_id}.watch-{artifact}.log"
    ).resolve()


def raise_if_watch_exited_early(
    *,
    process: _WatchProcess,
    log_path: Path,
) -> None:
    try:
        return_code = process.wait(timeout=WATCH_STARTUP_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        return

    if return_code == 0:
        return

    detail = watch_failure_detail(log_path)
    if detail is None:
        raise RuntimeError(f"watch exited immediately with code {return_code}; see {log_path}")
    raise RuntimeError(f"watch exited immediately with code {return_code}: {detail}")


def reap_watch_when_done(
    *,
    db_path: Path,
    process: _WatchProcess,
    lease_id: str,
    run_id: str,
    artifact: str,
    log_path: Path,
) -> None:
    """Reap a watch process and persist abnormal exits for the run UI."""

    thread = threading.Thread(
        target=_watch_reaper,
        kwargs={
            "db_path": db_path,
            "process": process,
            "lease_id": lease_id,
            "run_id": run_id,
            "artifact": artifact,
            "log_path": log_path,
        },
        name=f"run-manager-reap-watch-{process.pid}",
        daemon=True,
    )
    thread.start()


def _watch_reaper(
    *,
    db_path: Path,
    process: _WatchProcess,
    lease_id: str,
    run_id: str,
    artifact: str,
    log_path: Path,
) -> None:
    return_code = process.wait()
    store = ManagerStore(db_path)
    store.clear_viewer_lease(lease_id=lease_id, pid=process.pid)
    if return_code == 0:
        return
    store.append_run_event(
        run_id=run_id,
        kind="watch_failed",
        message=_watch_failure_event_message(
            artifact=artifact,
            return_code=return_code,
            log_path=log_path,
        ),
    )


def _watch_failure_event_message(
    *,
    artifact: str,
    return_code: int,
    log_path: Path,
) -> str:
    detail = watch_failure_detail(log_path)
    if detail is None:
        return f"{artifact} watch exited with code {return_code}; see {log_path}"
    return f"{artifact} watch failed: {detail}"


def watch_failure_detail(log_path: Path) -> str | None:
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        detail = line.strip()
        if _is_traceback_exception_line(detail):
            return detail
    for line in reversed(lines):
        detail = line.strip()
        if detail and not detail.startswith("#"):
            return detail
    return None


def _is_traceback_exception_line(detail: str) -> bool:
    return detail.startswith(
        (
            "RuntimeError:",
            "ValueError:",
            "FileNotFoundError:",
            "torch.OutOfMemoryError:",
        )
    )


def active_watch_pid(
    *,
    store: ManagerStore,
    lease_id: str,
    run_id: str,
    artifact: str,
) -> int | None:
    lease = store.get_viewer_lease(lease_id)
    if lease is None:
        return None
    if lease.kind != "run_watch" or lease.owner_id != run_id or lease.qualifier != artifact:
        store.clear_viewer_lease(lease_id=lease_id)
        return None
    if not viewer_lease_is_fresh(lease):
        store.clear_viewer_lease(lease_id=lease_id, pid=lease.pid)
        return None
    if watch_process_matches(
        pid=lease.pid,
        run_id=run_id,
        artifact=artifact,
    ):
        return lease.pid
    store.clear_viewer_lease(lease_id=lease_id, pid=lease.pid)
    return None


def watch_process_matches(*, pid: int, run_id: str, artifact: str) -> bool:
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
        and f"--run-id {run_id}" in normalized
    )
