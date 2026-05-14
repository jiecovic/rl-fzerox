# src/rl_fzerox/apps/run_manager/api/live.py
from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from contextlib import suppress

from fastapi import WebSocket, WebSocketDisconnect

RunSnapshotPayload = dict[str, list[dict[str, object]]]
RunSnapshotUpdate = RunSnapshotPayload | None


class RunLiveBroadcaster:
    """Share one live run snapshot poller across all connected WebSocket clients."""

    def __init__(
        self,
        load_snapshot: Callable[[], Awaitable[RunSnapshotPayload]],
        *,
        interval_seconds: float = 2.0,
    ) -> None:
        self._load_snapshot = load_snapshot
        self._interval_seconds = interval_seconds
        self._subscribers: set[asyncio.Queue[RunSnapshotUpdate]] = set()
        self._last_snapshot_key: str | None = None
        self._last_snapshot: RunSnapshotPayload | None = None
        self._poll_task: asyncio.Task[None] | None = None

    async def serve(self, websocket: WebSocket) -> None:
        await websocket.accept()
        queue: asyncio.Queue[RunSnapshotUpdate] = asyncio.Queue(maxsize=1)
        already_polling = len(self._subscribers) > 0
        if not already_polling:
            self._last_snapshot_key = None
            self._last_snapshot = None
        self._subscribers.add(queue)
        if already_polling and self._last_snapshot is not None:
            _queue_latest(queue, self._last_snapshot)
        self._ensure_polling()
        try:
            while True:
                snapshot_task = asyncio.create_task(queue.get())
                disconnect_task = asyncio.create_task(websocket.receive_text())
                done, pending = await asyncio.wait(
                    {snapshot_task, disconnect_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in pending:
                    task.cancel()
                for task in pending:
                    with suppress(asyncio.CancelledError):
                        await task
                if disconnect_task in done:
                    disconnect_task.result()
                    continue
                snapshot = snapshot_task.result()
                if snapshot is None:
                    await websocket.close(code=1011)
                    return
                await websocket.send_json({"type": "runs_snapshot", **snapshot})
        except (RuntimeError, WebSocketDisconnect):
            return
        finally:
            self._subscribers.discard(queue)

    def _ensure_polling(self) -> None:
        if self._poll_task is None or self._poll_task.done():
            self._poll_task = asyncio.create_task(self._poll_snapshots())

    async def _poll_snapshots(self) -> None:
        while self._subscribers:
            try:
                snapshot = await self._load_snapshot()
            except Exception:
                for queue in tuple(self._subscribers):
                    _queue_latest(queue, None)
                await asyncio.sleep(self._interval_seconds)
                continue
            snapshot_key = json.dumps(snapshot, sort_keys=True, separators=(",", ":"))
            if snapshot_key != self._last_snapshot_key:
                self._last_snapshot_key = snapshot_key
                self._last_snapshot = snapshot
                for queue in tuple(self._subscribers):
                    _queue_latest(queue, snapshot)
            await asyncio.sleep(self._interval_seconds)


def _queue_latest(
    queue: asyncio.Queue[RunSnapshotUpdate],
    snapshot: RunSnapshotUpdate,
) -> None:
    if queue.full():
        with suppress(asyncio.QueueEmpty):
            queue.get_nowait()
    queue.put_nowait(snapshot)
