# src/rl_fzerox/apps/run_manager/worker/cli.py
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from rl_fzerox.apps.run_manager.training_monitor import (
    RunControlSignal,
    build_manager_training_callback,
)
from rl_fzerox.apps.run_manager.worker.clock import now_iso, past_tense_command
from rl_fzerox.apps.run_manager.worker.config import (
    _resolved_train_config,
    _run_paths,
    _track_sampling_runtime_persistence,
)
from rl_fzerox.apps.run_manager.worker.heartbeat import (
    _heartbeat_or_die,
    _startup_reporter,
    _WorkerHeartbeatLoop,
)
from rl_fzerox.core.manager import ManagerStore, default_manager_db_path
from rl_fzerox.core.training.runner import run_training

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
            stopped_at=now_iso(),
            message=f"training {past_tense_command(signal.command)} by manager",
        )
        return
    except Exception as exc:
        LOGGER.exception("training failed for run_id=%s", run.id)
        store.update_run_status(
            run_id=run.id,
            status="failed",
            stopped_at=now_iso(),
            message=f"training failed: {type(exc).__name__}: {exc}",
        )
        raise
    else:
        LOGGER.info("marking run finished run_id=%s", run.id)
        store.update_run_status(
            run_id=run.id,
            status="finished",
            stopped_at=now_iso(),
            message="training finished",
        )
    finally:
        if heartbeat_loop is not None:
            heartbeat_loop.stop()
        store.clear_run_worker(run.id, launch_token=args.launch_token)


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


def _mark_worker_boot_failure(
    *,
    store: ManagerStore,
    run_id: str,
    launch_token: str,
    message: str,
) -> None:
    """Record failures that happen before the run config can be validated."""

    store.mark_worker_boot_failure(
        run_id=run_id,
        launch_token=launch_token,
        message=message,
        failed_at=now_iso(),
    )


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
