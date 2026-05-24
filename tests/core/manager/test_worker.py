# tests/core/manager/test_worker.py
from __future__ import annotations

from pathlib import Path

from rl_fzerox.apps.run_manager.worker import _mark_worker_boot_failure
from rl_fzerox.core.manager import ManagerStore, default_managed_run_config


def test_worker_boot_failure_marks_run_failed_without_loading_config(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "runs.db")
    run = store.create_run(
        run_id="boot-fail",
        name="Boot fail",
        config=default_managed_run_config(),
        managed_runs_root=tmp_path / "runs",
    )
    launch_token = "worker-token"
    store.register_run_worker(
        run_id=run.id,
        launch_token=launch_token,
        pid=123,
        launched_at="2026-05-09T00:00:00+00:00",
    )

    _mark_worker_boot_failure(
        store=store,
        run_id=run.id,
        launch_token=launch_token,
        message="boot exploded",
    )

    failed = store.get_run(run.id)
    assert failed is not None
    assert failed.status == "failed"
    assert store.pending_run_command(run.id) is None
    assert (
        store.heartbeat_run_worker(
            run_id=run.id,
            launch_token=launch_token,
            heartbeat_at="2026-05-09T00:00:01+00:00",
        )
        is False
    )
    events = store.list_recent_run_events((run.id,))[run.id]
    assert events[0].kind == "failed"
    assert events[0].message == "boot exploded"
