# src/rl_fzerox/apps/run_manager/worker.py
from __future__ import annotations

import argparse
import logging
import threading
from datetime import UTC, datetime
from pathlib import Path

from rl_fzerox.apps.run_manager.training_monitor import (
    RunControlSignal,
    build_manager_training_callback,
)
from rl_fzerox.core.manager import ManagerStore, default_manager_db_path
from rl_fzerox.core.manager.models import ManagedRun
from rl_fzerox.core.manager.projection.x_cup_runtime import (
    restore_generated_x_cup_entries_from_state,
)
from rl_fzerox.core.manager.registry.runs.maintenance import RUN_WORKER_LEASE_POLICY
from rl_fzerox.core.manager.training import (
    build_managed_fork_train_app_config,
    build_managed_resume_train_app_config,
    build_managed_train_app_config,
)
from rl_fzerox.core.training.runner import run_training
from rl_fzerox.core.training.runs import RUN_LAYOUT, RunPaths, continue_run_paths
from rl_fzerox.core.training.session.callbacks.track_sampling import (
    TrackSamplingRuntimePersistence,
)

LOGGER = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    """Run one manager-launched training job and write status back to SQLite."""

    _configure_logging()
    args = parse_args(argv)
    LOGGER.info(
        "manager worker booting run_id=%s resume=%s db_path=%s",
        args.run_id,
        args.resume,
        args.db_path,
    )
    store = ManagerStore(args.db_path)
    try:
        run = store.get_run(args.run_id)
    except Exception as exc:
        LOGGER.exception("manager worker failed before loading run_id=%s", args.run_id)
        _mark_worker_boot_failure(
            store=store,
            run_id=args.run_id,
            launch_token=args.launch_token,
            message=f"manager worker failed before loading run: {type(exc).__name__}: {exc}",
        )
        raise
    if run is None:
        LOGGER.error("managed run not found: %s", args.run_id)
        _mark_worker_boot_failure(
            store=store,
            run_id=args.run_id,
            launch_token=args.launch_token,
            message=f"managed run not found: {args.run_id}",
        )
        raise SystemExit(f"managed run not found: {args.run_id}")

    heartbeat_loop: _WorkerHeartbeatLoop | None = None
    try:
        _heartbeat_or_die(store=store, run_id=run.id, launch_token=args.launch_token)
        heartbeat_loop = _WorkerHeartbeatLoop(
            store=store,
            run_id=run.id,
            launch_token=args.launch_token,
        )
        heartbeat_loop.start()
        LOGGER.info(
            "loaded run status=%s run_dir=%s pending_command=%s",
            run.status,
            run.run_dir,
            run.pending_command,
        )
        _heartbeat_or_die(store=store, run_id=run.id, launch_token=args.launch_token)
        train_config = _resolved_train_config(
            store=store,
            run=run,
            resume=args.resume,
        )
        run_paths = _run_paths(run, resume=args.resume)
        _heartbeat_or_die(store=store, run_id=run.id, launch_token=args.launch_token)
        LOGGER.info(
            "built train config total_timesteps=%s explicit_run_dir=%s continue_run_dir=%s",
            train_config.train.total_timesteps,
            train_config.train.explicit_run_dir,
            train_config.train.continue_run_dir,
        )
        LOGGER.info(
            "resolved run paths runtime_root=%s checkpoints_dir=%s baseline_state_path=%s",
            run_paths.runtime_root,
            run_paths.checkpoints_dir,
            run_paths.baseline_state_path,
        )
        LOGGER.info("starting run_training for run_id=%s", run.id)
        run_training(
            train_config,
            track_sampling_runtime_persistence=_track_sampling_runtime_persistence(
                store=store,
                run_id=run.id,
            ),
            extra_callbacks=(
                build_manager_training_callback(
                    store=store,
                    run_id=run.id,
                    launch_token=args.launch_token,
                    run_paths=run_paths,
                    total_timesteps=train_config.train.total_timesteps,
                    lineage_step_offset=run.lineage_step_offset,
                ),
            ),
            startup_reporter=_startup_reporter(
                store=store,
                run_id=run.id,
                launch_token=args.launch_token,
            ),
        )
        LOGGER.info("run_training returned normally for run_id=%s", run.id)
    except RunControlSignal as signal:
        LOGGER.info("manager requested controlled %s for run_id=%s", signal.command, run.id)
        store.clear_run_command(run.id, command=signal.command)
        store.update_run_status(
            run_id=run.id,
            status="paused" if signal.command == "pause" else "stopped",
            stopped_at=_now(),
            message=f"training {_past_tense_command(signal.command)} by manager",
        )
        return
    except Exception as exc:
        LOGGER.exception("training failed for run_id=%s", run.id)
        store.update_run_status(
            run_id=run.id,
            status="failed",
            stopped_at=_now(),
            message=f"training failed: {type(exc).__name__}: {exc}",
        )
        raise
    else:
        LOGGER.info("marking run finished run_id=%s", run.id)
        store.update_run_status(
            run_id=run.id,
            status="finished",
            stopped_at=_now(),
            message="training finished",
        )
    finally:
        if heartbeat_loop is not None:
            heartbeat_loop.stop()
        store.clear_run_worker(run.id, launch_token=args.launch_token)


def _track_sampling_runtime_persistence(
    *,
    store: ManagerStore,
    run_id: str,
) -> TrackSamplingRuntimePersistence:
    return TrackSamplingRuntimePersistence(
        load=lambda: store.get_run_track_sampling_state(run_id),
        save=lambda state: store.upsert_run_track_sampling_state(
            run_id=run_id,
            state=state,
            updated_at=_now(),
        ),
        replace_materialized_artifacts=lambda artifacts: (
            store.replace_run_track_sampling_artifacts(
                run_id=run_id,
                artifacts=artifacts,
            )
        ),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one manager-launched training job")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--launch-token", required=True)
    parser.add_argument(
        "--db-path",
        type=Path,
        default=default_manager_db_path(),
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def _resolved_train_config(*, store: ManagerStore, run: ManagedRun, resume: bool):
    if resume:
        config = build_managed_resume_train_app_config(
            run.config,
            run_id=run.id,
            run_dir=run.run_dir,
            tensorboard_step_offset=run.lineage_step_offset,
        )
        return restore_generated_x_cup_entries_from_state(
            config,
            state=store.get_run_track_sampling_state(run.id),
        )
    if run.source_snapshot_dir is not None and run.source_artifact is not None:
        if run.source_run_id is None:
            raise RuntimeError("managed fork source metadata is missing; recreate the fork draft")
        source_run = store.get_run(run.source_run_id)
        if source_run is None:
            raise RuntimeError(f"source run not found for forked run: {run.source_run_id}")
        return build_managed_fork_train_app_config(
            run.config,
            run_id=run.id,
            run_dir=run.run_dir,
            source_run_dir=run.source_snapshot_dir,
            source_artifact=run.source_artifact,
            source_config=source_run.config,
            tensorboard_step_offset=run.lineage_step_offset,
        )
    if run.source_run_id is not None and run.source_artifact is not None:
        source_run = store.get_run(run.source_run_id)
        if source_run is None:
            raise RuntimeError(f"source run not found for forked run: {run.source_run_id}")
        return build_managed_fork_train_app_config(
            run.config,
            run_id=run.id,
            run_dir=run.run_dir,
            source_run_dir=source_run.run_dir,
            source_artifact=run.source_artifact,
            source_config=source_run.config,
            tensorboard_step_offset=run.lineage_step_offset,
        )
    return build_managed_train_app_config(
        run.config,
        run_id=run.id,
        run_dir=run.run_dir,
    )


def _run_paths(run: ManagedRun, *, resume: bool) -> RunPaths:
    if resume:
        return continue_run_paths(run.run_dir)
    return RunPaths(
        run_dir=run.run_dir,
        fresh_run=True,
        runtime_root=run.run_dir / RUN_LAYOUT.runtime_dirname,
        tensorboard_dir=run.run_dir / RUN_LAYOUT.tensorboard_dirname,
        checkpoints_dir=run.run_dir / RUN_LAYOUT.checkpoints_dirname,
        track_sampling_state_path=run.run_dir
        / RUN_LAYOUT.runtime_dirname
        / RUN_LAYOUT.track_sampling_state_filename,
        latest_model_path=run.run_dir / RUN_LAYOUT.model_artifacts.latest,
        latest_policy_path=run.run_dir / RUN_LAYOUT.policy_artifacts.latest,
        best_model_path=run.run_dir / RUN_LAYOUT.model_artifacts.best,
        best_policy_path=run.run_dir / RUN_LAYOUT.policy_artifacts.best,
        final_model_path=run.run_dir / RUN_LAYOUT.model_artifacts.final,
        final_policy_path=run.run_dir / RUN_LAYOUT.policy_artifacts.final,
        baselines_dir=run.run_dir / RUN_LAYOUT.baselines_dirname,
        baseline_state_path=run.run_dir
        / RUN_LAYOUT.baselines_dirname
        / RUN_LAYOUT.baseline_filename,
    )


def _mark_worker_boot_failure(
    *,
    store: ManagerStore,
    run_id: str,
    launch_token: str,
    message: str,
) -> None:
    """Record failures that happen before the run config can be validated."""

    store._ensure_schema_initialized()
    with store._connect() as connection:
        row = connection.execute("SELECT id FROM runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            return
        connection.execute("DELETE FROM run_commands WHERE run_id = ?", (run_id,))
        connection.execute(
            "DELETE FROM run_workers WHERE run_id = ? AND launch_token = ?",
            (run_id, launch_token),
        )
        connection.execute(
            """
            UPDATE runs
            SET status = ?, stopped_at = ?
            WHERE id = ?
            """,
            ("failed", _now(), run_id),
        )
        connection.execute(
            """
            INSERT INTO run_events(run_id, created_at, kind, message)
            VALUES (?, ?, ?, ?)
            """,
            (run_id, _now(), "failed", message),
        )


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _past_tense_command(command: str) -> str:
    return "stopped" if command == "stop" else f"{command}d"


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _startup_reporter(*, store: ManagerStore, run_id: str, launch_token: str):
    def report(kind: str, message: str) -> None:
        _heartbeat_or_die(store=store, run_id=run_id, launch_token=launch_token)
        pending_command = store.pending_run_command(run_id)
        if pending_command is not None:
            raise RunControlSignal(pending_command)
        store.append_run_event(
            run_id=run_id,
            kind=kind,
            message=message,
        )

    return report


def _heartbeat_or_die(*, store: ManagerStore, run_id: str, launch_token: str) -> None:
    heartbeat_ok = store.heartbeat_run_worker(
        run_id=run_id,
        launch_token=launch_token,
        heartbeat_at=_now(),
    )
    if not heartbeat_ok:
        raise RuntimeError("manager worker lease is missing or stale")


class _WorkerHeartbeatLoop:
    def __init__(
        self,
        *,
        store: ManagerStore,
        run_id: str,
        launch_token: str,
        interval_seconds: float | None = None,
    ) -> None:
        self._store = store
        self._run_id = run_id
        self._launch_token = launch_token
        self._interval_seconds = (
            RUN_WORKER_LEASE_POLICY.heartbeat_interval.total_seconds()
            if interval_seconds is None
            else interval_seconds
        )
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name=f"manager-heartbeat-{run_id}",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=self._interval_seconds + 1.0)

    def _run(self) -> None:
        while not self._stop_event.wait(self._interval_seconds):
            heartbeat_ok = self._store.heartbeat_run_worker(
                run_id=self._run_id,
                launch_token=self._launch_token,
                heartbeat_at=_now(),
            )
            if heartbeat_ok:
                continue
            LOGGER.error(
                "manager worker lease lost in background heartbeat run_id=%s",
                self._run_id,
            )
            return


if __name__ == "__main__":
    main()
