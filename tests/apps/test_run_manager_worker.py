# tests/apps/test_run_manager_worker.py
from __future__ import annotations

from pathlib import Path

import pytest

from rl_fzerox.apps.run_manager.training_monitor import RunControlSignal
from rl_fzerox.apps.run_manager.worker import _startup_reporter
from rl_fzerox.core.manager import ManagerStore, default_managed_run_config


def test_startup_reporter_raises_for_pending_manager_stop(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    run = store.create_run(
        run_id="run-001",
        name="Worker Run",
        config=default_managed_run_config(),
        explicit_run_dir=tmp_path / "runs" / "run-001",
    )
    launched_at = "2026-05-05T10:00:00+00:00"
    store.update_run_status(
        run_id=run.id,
        status="running",
        started_at=launched_at,
        stopped_at=None,
        message="worker launched",
    )
    assert store.register_run_worker(
        run_id=run.id,
        launch_token="token-1",
        pid=12345,
        launched_at=launched_at,
    )
    store.request_run_command(run_id=run.id, command="stop")

    reporter = _startup_reporter(store=store, run_id=run.id, launch_token="token-1")

    with pytest.raises(RunControlSignal, match="stop"):
        reporter("startup_materialize", "Materializing track sampling baselines: 0/2 complete")
