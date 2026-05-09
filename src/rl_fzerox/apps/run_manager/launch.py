# src/rl_fzerox/apps/run_manager/launch.py
from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4

from rl_fzerox.apps.watch_cli.resolve import resolve_watch_app_config
from rl_fzerox.core.manager import ManagedRun, ManagedRunConfig, ManagerStore, new_managed_run_id
from rl_fzerox.core.manager.artifacts.fork_source import (
    clone_fork_source,
    is_complete_fork_source,
    run_fork_source_dir,
    snapshot_fork_source,
)
from rl_fzerox.core.manager.artifacts.paths import predicted_managed_run_dir
from rl_fzerox.core.manager.models import RunCommand
from rl_fzerox.core.manager.training import (
    assert_managed_fork_compatible,
    build_managed_fork_train_app_config,
    build_managed_train_app_config,
)
from rl_fzerox.core.runtime_spec.paths import project_root_dir
from rl_fzerox.core.runtime_spec.schema import TrainAppConfig
from rl_fzerox.core.training.runs import (
    resolve_model_artifact_path,
    save_train_run_config,
)


class ManagerRunLauncher:
    """Launch and control manager-owned training runs."""

    def __init__(self, store: ManagerStore) -> None:
        self._store = store

    def launch(
        self,
        *,
        name: str,
        config: ManagedRunConfig,
        draft_id: str | None = None,
        source_run_id: str | None = None,
        source_artifact: Literal["latest", "best"] | None = None,
    ) -> ManagedRun:
        normalized_name = name.strip()
        if not normalized_name:
            raise ValueError("run name is required")
        if source_run_id is not None and source_artifact is not None:
            if draft_id is None:
                return self.fork(
                    run_id=source_run_id,
                    artifact=source_artifact,
                    name=normalized_name,
                    config=config,
                )
            draft = self._store.get_draft(draft_id)
            if draft is None:
                raise ValueError(f"draft not found: {draft_id}")
            if draft.source_run_id != source_run_id or draft.source_artifact != source_artifact:
                raise ValueError("fork draft source no longer matches the requested source run")
            if draft.source_snapshot_dir is None or draft.source_num_timesteps is None:
                raise ValueError(
                    "fork draft is missing its pinned checkpoint snapshot; recreate the fork draft"
                )
            return self.fork(
                run_id=source_run_id,
                artifact=source_artifact,
                name=normalized_name,
                config=config,
                exclude_draft_id=draft_id,
                source_snapshot_dir=draft.source_snapshot_dir,
                source_num_timesteps=draft.source_num_timesteps,
            )

        run_id = new_managed_run_id(normalized_name)
        run_dir = predicted_managed_run_dir(run_id, lineage_id=run_id)
        train_config = build_managed_train_app_config(
            config,
            run_id=run_id,
            run_dir=run_dir,
        )
        run = self._store.create_run(
            run_id=run_id,
            name=normalized_name,
            config=config,
            explicit_run_dir=run_dir,
            lineage_id=run_id,
            exclude_draft_id=draft_id,
        )
        _persist_launch_manifest(run_dir=run.run_dir, train_config=train_config)
        self._spawn_worker(run_id=run.id, resume=False)
        launched = self._store.update_run_status(
            run_id=run.id,
            status="running",
            started_at=_utc_now(),
            stopped_at=None,
            message=f"training worker launched; log: {_manager_worker_log_path(run.id)}",
        )
        if launched is None:
            raise RuntimeError(f"managed run disappeared during launch: {run.id}")
        return launched

    def fork(
        self,
        *,
        run_id: str,
        artifact: Literal["latest", "best"],
        name: str | None = None,
        config: ManagedRunConfig | None = None,
        exclude_draft_id: str | None = None,
        source_snapshot_dir: Path | None = None,
        source_num_timesteps: int | None = None,
    ) -> ManagedRun:
        """Launch one child run warm-started from a parent run checkpoint."""

        source_run = self._store.get_run(run_id)
        if source_run is None:
            raise ValueError(f"run not found: {run_id}")
        if artifact not in {"latest", "best"}:
            raise ValueError(f"unsupported fork artifact: {artifact}")

        normalized_name = (name or _default_fork_name(source_run.name, artifact)).strip()
        if not normalized_name:
            raise ValueError("run name is required")
        child_config = config or source_run.config
        assert_managed_fork_compatible(source_run.config, child_config)
        child_run_id = new_managed_run_id(normalized_name)
        child_run_dir = predicted_managed_run_dir(
            child_run_id,
            lineage_id=source_run.lineage_id,
        )
        child_source_snapshot_dir = run_fork_source_dir(run_dir=child_run_dir)
        if source_snapshot_dir is None or source_num_timesteps is None:
            source_num_timesteps = snapshot_fork_source(
                source_run_dir=source_run.run_dir,
                artifact=artifact,
                destination_dir=child_source_snapshot_dir,
            )
        else:
            clone_fork_source(
                source_dir=source_snapshot_dir,
                destination_dir=child_source_snapshot_dir,
            )
        child_lineage_step_offset = source_run.lineage_step_offset + source_num_timesteps
        train_config = build_managed_fork_train_app_config(
            child_config,
            run_id=child_run_id,
            run_dir=child_run_dir,
            source_run_dir=child_source_snapshot_dir,
            source_artifact=artifact,
            tensorboard_step_offset=child_lineage_step_offset,
        )
        child_run = self._store.create_run(
            run_id=child_run_id,
            name=normalized_name,
            config=child_config,
            explicit_run_dir=child_run_dir,
            lineage_id=source_run.lineage_id,
            lineage_step_offset=child_lineage_step_offset,
            parent_run_id=source_run.id,
            source_run_id=source_run.id,
            source_artifact=artifact,
            source_snapshot_dir=child_source_snapshot_dir,
            source_num_timesteps=source_num_timesteps,
            exclude_draft_id=exclude_draft_id,
        )
        _persist_launch_manifest(run_dir=child_run.run_dir, train_config=train_config)
        self._spawn_worker(run_id=child_run.id, resume=False)
        launched = self._store.update_run_status(
            run_id=child_run.id,
            status="running",
            started_at=_utc_now(),
            stopped_at=None,
            message=(
                f"forked from {source_run.name} ({artifact} @ {source_num_timesteps:,} steps); "
                f"log: {_manager_worker_log_path(child_run.id)}"
            ),
        )
        if launched is None:
            raise RuntimeError(f"managed child run disappeared during launch: {child_run.id}")
        return launched

    def resume(self, *, run_id: str) -> ManagedRun:
        """Resume one paused or stopped run in place from its latest checkpoint."""

        run = self._store.get_run(run_id)
        if run is None:
            raise ValueError(f"run not found: {run_id}")
        if run.status not in {"paused", "stopped", "failed"}:
            raise ValueError("only paused, stopped, or failed runs can be resumed")

        self._store.clear_run_command(run.id)
        reset_local_clock = False
        try:
            resolve_model_artifact_path(run.run_dir, artifact="latest")
        except FileNotFoundError:
            if run.source_snapshot_dir is None and run.source_run_id is None:
                raise ValueError(
                    "no resumable checkpoint exists for this run yet; "
                    "resume cannot continue it safely"
                ) from None
            run, restored = self._ensure_fork_source_snapshot(run)
            reset_local_clock = True
            self._spawn_worker(run_id=run.id, resume=False)
            message = (
                "training worker relaunched from "
                f"{'rebuilt ' if restored else ''}pinned fork source; "
                f"log: {_manager_worker_log_path(run.id)}"
            )
        else:
            self._spawn_worker(run_id=run.id, resume=True)
            message = (
                "training worker resumed from latest checkpoint; "
                f"log: {_manager_worker_log_path(run.id)}"
            )

        if reset_local_clock:
            self._store.clear_run_runtime(run.id)
        resumed = self._store.update_run_status(
            run_id=run.id,
            status="running",
            started_at=_utc_now() if reset_local_clock else None,
            stopped_at=None,
            message=message,
        )
        if resumed is None:
            raise RuntimeError(f"managed run disappeared during resume: {run.id}")
        return resumed

    def _ensure_fork_source_snapshot(self, run: ManagedRun) -> tuple[ManagedRun, bool]:
        """Restore a missing or incomplete pinned fork source before warm start."""

        snapshot_dir = run.source_snapshot_dir or run_fork_source_dir(run_dir=run.run_dir)
        if run.source_artifact is not None and is_complete_fork_source(
            source_dir=snapshot_dir,
            artifact=run.source_artifact,
        ):
            return run, False
        if run.source_run_id is None or run.source_artifact is None:
            raise ValueError(
                "fork source snapshot is missing and this run cannot rebuild it safely"
            )
        source_run = self._store.get_run(run.source_run_id)
        if source_run is None:
            raise ValueError(f"source run not found for forked run: {run.source_run_id}")
        source_num_timesteps = snapshot_fork_source(
            source_run_dir=source_run.run_dir,
            artifact=run.source_artifact,
            destination_dir=snapshot_dir,
        )
        refreshed = self._store.update_run_fork_source(
            run_id=run.id,
            source_snapshot_dir=snapshot_dir,
            source_num_timesteps=source_num_timesteps,
            lineage_step_offset=source_run.lineage_step_offset + source_num_timesteps,
        )
        if refreshed is None:
            raise RuntimeError(f"managed run disappeared while rebuilding fork source: {run.id}")
        return refreshed, True

    def request_pause(self, *, run_id: str) -> ManagedRun:
        """Request a graceful pause for one running run."""

        return self._request_command(run_id=run_id, command="pause")

    def request_stop(self, *, run_id: str) -> ManagedRun:
        """Request a graceful stop for one running run."""

        return self._request_command(run_id=run_id, command="stop")

    def watch_artifact(self, *, run_id: str, artifact: str) -> WatchLaunchStatus:
        """Launch the desktop watch app against one saved artifact for one run."""

        run = self._store.get_run(run_id)
        if run is None:
            raise ValueError(f"run not found: {run_id}")
        if artifact not in {"latest", "best"}:
            raise ValueError(f"unsupported watch artifact: {artifact}")
        resolve_model_artifact_path(run.run_dir, artifact=artifact)
        pid_path = _manager_watch_pid_path(run.id, artifact=artifact)
        if _active_watch_pid(
            pid_path=pid_path,
            run_id=run.id,
            run_dir=run.run_dir,
            artifact=artifact,
        ) is not None:
            return "already_running"
        resolve_watch_app_config(
            policy_run_dir=None,
            policy_artifact="best" if artifact == "best" else "latest",
            manager_db_path=self._store.db_path,
            managed_run_id=run.id,
            overrides=(),
        )
        log_path = _manager_watch_log_path(run.id, artifact=artifact)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "-m",
            "rl_fzerox.apps.watch",
            "--manager-db-path",
            str(self._store.db_path),
            "--managed-run-id",
            run.id,
            "--artifact",
            artifact,
            "--watch-pid-file",
            str(pid_path),
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
        _write_watch_pid_file(
            pid_path=pid_path,
            pid=process.pid,
            run_id=run.id,
            run_dir=run.run_dir,
            artifact=artifact,
        )
        _raise_if_watch_exited_early(process=process, log_path=log_path, pid_path=pid_path)
        return "started"

    def _request_command(self, *, run_id: str, command: RunCommand) -> ManagedRun:
        run = self._store.get_run(run_id)
        if run is None:
            raise ValueError(f"run not found: {run_id}")
        if run.status != "running":
            raise ValueError("only running runs can be controlled")
        updated = self._store.request_run_command(run_id=run_id, command=command)
        if updated is None:
            raise RuntimeError(f"managed run disappeared during {command}: {run_id}")
        return updated

    def _spawn_worker(self, *, run_id: str, resume: bool) -> None:
        log_path = _manager_worker_log_path(run_id)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        launch_token = uuid4().hex
        command = [
            sys.executable,
            "-m",
            "rl_fzerox.apps.run_manager.worker",
            "--db-path",
            str(self._store.db_path),
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
                registered = self._store.register_run_worker(
                    run_id=run_id,
                    launch_token=launch_token,
                    pid=process.pid,
                    launched_at=_utc_now(),
                )
                if not registered:
                    raise RuntimeError(
                        f"managed run disappeared before worker registration: {run_id}"
                    )
            except Exception:
                self._store.clear_run_worker(run_id)
                self._store.update_run_status(
                    run_id=run_id,
                    status="failed",
                    stopped_at=_utc_now(),
                    message=f"failed to launch manager worker; see {log_path}",
                )
                raise


def _manager_worker_log_path(run_id: str) -> Path:
    return (project_root_dir() / "local" / "manager" / "logs" / f"{run_id}.log").resolve()


WatchLaunchStatus = Literal["started", "already_running"]


def _manager_watch_log_path(run_id: str, *, artifact: str) -> Path:
    return (
        project_root_dir() / "local" / "manager" / "logs" / f"{run_id}.watch-{artifact}.log"
    ).resolve()


def _manager_watch_pid_path(run_id: str, *, artifact: str) -> Path:
    return (
        project_root_dir() / "local" / "manager" / "watch" / f"{run_id}.watch-{artifact}.json"
    ).resolve()


def _persist_launch_manifest(*, run_dir: Path, train_config: TrainAppConfig) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    save_train_run_config(config=train_config, run_dir=run_dir)


def _default_fork_name(source_name: str, artifact: Literal["latest", "best"]) -> str:
    suffix = "best fork" if artifact == "best" else "fork"
    return f"{source_name} {suffix}"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _raise_if_watch_exited_early(
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
    detail = _watch_failure_detail(log_path)
    if detail is None:
        raise RuntimeError(f"watch exited immediately with code {return_code}; see {log_path}")
    raise RuntimeError(f"watch exited immediately with code {return_code}: {detail}")


def _watch_failure_detail(log_path: Path) -> str | None:
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        detail = line.strip()
        if detail:
            return detail
    return None


def _active_watch_pid(*, pid_path: Path, run_id: str, run_dir: Path, artifact: str) -> int | None:
    payload = _read_watch_pid_file(pid_path)
    if payload is None:
        return None
    pid = payload.get("pid")
    if not isinstance(pid, int):
        pid_path.unlink(missing_ok=True)
        return None
    if _watch_process_matches(pid=pid, run_id=run_id, run_dir=run_dir, artifact=artifact):
        return pid
    pid_path.unlink(missing_ok=True)
    return None


def _write_watch_pid_file(
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


def _read_watch_pid_file(pid_path: Path) -> dict[str, object] | None:
    try:
        return json.loads(pid_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError):
        pid_path.unlink(missing_ok=True)
        return None


def _watch_process_matches(*, pid: int, run_id: str, run_dir: Path, artifact: str) -> bool:
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
        and (
            f"--managed-run-id {run_id}" in normalized
            or f"--run-dir {run_dir}" in normalized
        )
    )
