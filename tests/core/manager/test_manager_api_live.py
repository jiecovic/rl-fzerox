# tests/core/manager/test_manager_api_live.py
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rl_fzerox.apps.run_manager.api import create_manager_api_app
from rl_fzerox.core.manager import (
    ManagerStore,
    default_managed_run_config,
)
from tests.core.manager.manager_api_support import (
    _LauncherStub,
    _write_track_sampling_state,
)

pytestmark = pytest.mark.anyio


def test_manager_api_live_track_sampling_sends_initial_snapshot(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    run = store.create_run(
        run_id="run-with-live-track-pool",
        name="Live Track Pool Run",
        config=default_managed_run_config(),
        explicit_run_dir=tmp_path / "runs" / "run-with-live-track-pool",
    )
    store.update_run_status(
        run_id=run.id,
        status="running",
        started_at="2026-05-04T12:00:00+00:00",
        stopped_at=None,
        message="worker launched",
    )
    _write_track_sampling_state(store, run.id)

    app = create_manager_api_app(store, run_launcher=_LauncherStub())

    with TestClient(app) as client:
        with client.websocket_connect(f"/api/runs/{run.id}/track-sampling/live") as websocket:
            payload = websocket.receive_json()

    assert payload["type"] == "track_sampling_snapshot"
    assert payload["state"]["entries"][0]["label"] == "Mute City"


def test_manager_api_live_runs_sends_initial_snapshot(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    run = store.create_run(
        name="Visible Live Run",
        config=default_managed_run_config(),
        managed_runs_root=tmp_path / "runs",
    )
    store.update_run_status(run_id=run.id, status="stopped", message="run stopped")

    app = create_manager_api_app(store, run_launcher=_LauncherStub())

    with TestClient(app) as client:
        with client.websocket_connect("/api/runs/live") as websocket:
            payload = websocket.receive_json()

    assert payload["type"] == "runs_snapshot"
    assert payload["runs"][0]["id"] == run.id
