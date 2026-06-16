# tests/core/manager/test_manager_api_live.py
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import WebSocketDisconnect

from rl_fzerox.apps.run_manager.api import handlers
from rl_fzerox.apps.run_manager.api.execution import run_sync
from rl_fzerox.apps.run_manager.api.live import (
    KeyedLiveSnapshotBroadcaster,
    LiveMessageTypes,
    LiveSnapshotBroadcaster,
)
from rl_fzerox.core.manager import (
    ManagerStore,
    default_managed_run_config,
)
from tests.core.manager.manager_api_support import (
    _write_track_sampling_state,
)

pytestmark = pytest.mark.anyio


class _DisconnectingWebSocket:
    def __init__(self) -> None:
        self.accepted = False
        self.messages: list[dict[str, object]] = []

    async def accept(self) -> None:
        self.accepted = True

    async def send_json(self, data: dict[str, object]) -> None:
        self.messages.append(data)
        raise WebSocketDisconnect()


async def test_manager_api_live_track_sampling_sends_initial_snapshot(tmp_path: Path) -> None:
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

    broadcaster = KeyedLiveSnapshotBroadcaster(
        lambda run_id: run_sync(handlers.run_track_sampling_payload, store, run_id),
        message_types=LiveMessageTypes(
            snapshot="track_sampling_snapshot",
            error="track_sampling_error",
        ),
        error_log_message="failed to poll live track-pool snapshot",
    )
    websocket = _DisconnectingWebSocket()

    await broadcaster.serve(run.id, websocket)

    assert websocket.accepted is True
    payload = websocket.messages[0]
    assert payload["type"] == "track_sampling_snapshot"
    state = payload["state"]
    assert isinstance(state, dict)
    entries = state["entries"]
    assert isinstance(entries, list)
    first_entry = entries[0]
    assert isinstance(first_entry, dict)
    assert first_entry["label"] == "Mute City"


async def test_manager_api_live_runs_sends_initial_snapshot(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    run = store.create_run(
        name="Visible Live Run",
        config=default_managed_run_config(),
        managed_runs_root=tmp_path / "runs",
    )
    store.update_run_status(run_id=run.id, status="stopped", message="run stopped")

    broadcaster = LiveSnapshotBroadcaster(
        lambda: run_sync(handlers.runs_payload, store),
        message_types=LiveMessageTypes(snapshot="runs_snapshot", error="runs_error"),
        error_log_message="failed to poll live run snapshot",
    )
    websocket = _DisconnectingWebSocket()

    await broadcaster.serve(websocket)

    assert websocket.accepted is True
    payload = websocket.messages[0]
    assert payload["type"] == "runs_snapshot"
    runs = payload["runs"]
    assert isinstance(runs, list)
    first_run = runs[0]
    assert isinstance(first_run, dict)
    assert first_run["id"] == run.id
