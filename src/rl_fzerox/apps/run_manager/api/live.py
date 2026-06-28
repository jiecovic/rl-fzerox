# src/rl_fzerox/apps/run_manager/api/live.py
from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass

from fastapi import HTTPException, WebSocket, WebSocketDisconnect

LivePayload = Mapping[str, object]

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class LiveMessageTypes:
    snapshot: str
    error: str


@dataclass(frozen=True)
class LiveSnapshotUpdate:
    payload: LivePayload


@dataclass(frozen=True)
class LiveErrorUpdate:
    message: str


LiveUpdate = LiveSnapshotUpdate | LiveErrorUpdate


class LiveSnapshotBroadcaster:
    """Share one JSON snapshot poller across matching WebSocket clients."""

    def __init__(
        self,
        load_snapshot: Callable[[], Awaitable[LivePayload]],
        *,
        message_types: LiveMessageTypes,
        error_log_message: str,
        interval_seconds: float = 2.0,
    ) -> None:
        self._load_snapshot = load_snapshot
        self._message_types = message_types
        self._error_log_message = error_log_message
        self._interval_seconds = interval_seconds
        self._subscribers: set[asyncio.Queue[LiveUpdate]] = set()
        self._last_snapshot_key: str | None = None
        self._last_snapshot: LivePayload | None = None
        self._poll_task: asyncio.Task[None] | None = None

    @property
    def has_subscribers(self) -> bool:
        return len(self._subscribers) > 0

    async def serve(self, websocket: WebSocket) -> None:
        await websocket.accept()
        queue: asyncio.Queue[LiveUpdate] = asyncio.Queue(maxsize=1)
        already_polling = len(self._subscribers) > 0
        if not already_polling:
            self._last_snapshot_key = None
            self._last_snapshot = None
        self._subscribers.add(queue)
        if already_polling and self._last_snapshot is not None:
            _queue_latest(queue, LiveSnapshotUpdate(self._last_snapshot))
        self._ensure_polling()
        try:
            while True:
                update = await queue.get()
                if isinstance(update, LiveErrorUpdate):
                    await websocket.send_json(
                        {"type": self._message_types.error, "message": update.message}
                    )
                    continue
                await websocket.send_json({"type": self._message_types.snapshot, **update.payload})
        except (asyncio.CancelledError, RuntimeError, WebSocketDisconnect):
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
            except HTTPException as exc:
                message = f"{type(exc).__name__}: {exc.detail}"
                for queue in tuple(self._subscribers):
                    _queue_latest(queue, LiveErrorUpdate(message))
                if exc.status_code == 404:
                    return
                LOGGER.warning("%s: %s", self._error_log_message, message)
                await asyncio.sleep(self._interval_seconds)
                continue
            except Exception as exc:
                LOGGER.exception(self._error_log_message)
                message = f"{type(exc).__name__}: {exc}"
                for queue in tuple(self._subscribers):
                    _queue_latest(queue, LiveErrorUpdate(message))
                await asyncio.sleep(self._interval_seconds)
                continue
            snapshot_key = json.dumps(snapshot, sort_keys=True, separators=(",", ":"))
            if snapshot_key != self._last_snapshot_key:
                self._last_snapshot_key = snapshot_key
                self._last_snapshot = snapshot
                for queue in tuple(self._subscribers):
                    _queue_latest(queue, LiveSnapshotUpdate(snapshot))
            await asyncio.sleep(self._interval_seconds)


class KeyedLiveSnapshotBroadcaster:
    """Keep one live snapshot broadcaster per active subscription key."""

    def __init__(
        self,
        load_snapshot: Callable[[str], Awaitable[LivePayload]],
        *,
        message_types: LiveMessageTypes,
        error_log_message: str,
        interval_seconds: float = 2.0,
    ) -> None:
        self._load_snapshot = load_snapshot
        self._message_types = message_types
        self._error_log_message = error_log_message
        self._interval_seconds = interval_seconds
        self._broadcasters: dict[str, LiveSnapshotBroadcaster] = {}

    async def serve(self, key: str, websocket: WebSocket) -> None:
        broadcaster = self._broadcaster(key)
        try:
            await broadcaster.serve(websocket)
        finally:
            if not broadcaster.has_subscribers:
                self._broadcasters.pop(key, None)

    def _broadcaster(self, key: str) -> LiveSnapshotBroadcaster:
        broadcaster = self._broadcasters.get(key)
        if broadcaster is not None:
            return broadcaster
        broadcaster = LiveSnapshotBroadcaster(
            lambda: self._load_snapshot(key),
            message_types=self._message_types,
            error_log_message=self._error_log_message,
            interval_seconds=self._interval_seconds,
        )
        self._broadcasters[key] = broadcaster
        return broadcaster


def _queue_latest(
    queue: asyncio.Queue[LiveUpdate],
    snapshot: LiveUpdate,
) -> None:
    if queue.full():
        with suppress(asyncio.QueueEmpty):
            queue.get_nowait()
    queue.put_nowait(snapshot)
