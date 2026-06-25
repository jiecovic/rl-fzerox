# tests/ui/test_watch_runtime_worker_lifecycle.py
"""Watch runtime IPC tests for worker shutdown and bootstrap reporting.

The tests keep process lifecycle behavior separate from command coalescing, so
KeyboardInterrupt handling and bootstrap context reporting remain easy to audit.
"""

from __future__ import annotations

from pathlib import Path

from pytest import MonkeyPatch

from rl_fzerox.core.runtime_spec.schema import EmulatorConfig, WatchAppConfig
from rl_fzerox.ui.watch.runtime.ipc import ViewerCommand, WatchWorker, WorkerClosed, WorkerError
from rl_fzerox.ui.watch.runtime.live.worker import run_simulation_worker
from tests.ui.watch_runtime_ipc_support import _InterruptingProcess, _ShutdownQueue


def test_watch_worker_shutdown_swallows_keyboard_interrupt_during_join() -> None:
    command_queue = _ShutdownQueue()
    snapshot_queue = _ShutdownQueue()
    process = _InterruptingProcess()
    worker = WatchWorker(
        process=process,
        command_queue=command_queue,  # type: ignore[arg-type]  # queue test double
        snapshot_queue=snapshot_queue,  # type: ignore[arg-type]  # queue test double
    )

    worker.shutdown()

    assert len(command_queue.items) == 1
    command = command_queue.items[0]
    assert isinstance(command, ViewerCommand)
    assert command.quit_requested is True
    assert process.terminate_calls == 1
    assert command_queue.cancelled is True
    assert command_queue.closed is True
    assert snapshot_queue.cancelled is True
    assert snapshot_queue.closed is True


def test_run_simulation_worker_swallows_keyboard_interrupt(
    monkeypatch: MonkeyPatch,
) -> None:
    published: list[object] = []

    def _raise_keyboard_interrupt(*_args: object, **_kwargs: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(
        "rl_fzerox.ui.watch.runtime.live.worker._run_simulation_loop",
        _raise_keyboard_interrupt,
    )
    monkeypatch.setattr(
        "rl_fzerox.ui.watch.runtime.live.worker.publish_worker_message",
        lambda _queue, message: published.append(message),
    )

    run_simulation_worker(
        config=object(),  # type: ignore[arg-type]
        command_queue=object(),  # type: ignore[arg-type]
        snapshot_queue=object(),  # type: ignore[arg-type]
    )

    assert len(published) == 1
    assert isinstance(published[0], WorkerClosed)


def test_run_simulation_worker_reports_bootstrap_context(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    core_path = tmp_path / "core.so"
    rom_path = tmp_path / "fzerox.n64"
    runtime_dir = tmp_path / "runtime"
    core_path.write_bytes(b"core")
    rom_path.write_bytes(b"rom")
    config = WatchAppConfig(
        emulator=EmulatorConfig(
            core_path=core_path,
            rom_path=rom_path,
            runtime_dir=runtime_dir,
            renderer="gliden64",
        )
    )
    published: list[object] = []

    def _raise_bootstrap_error(_config: WatchAppConfig) -> None:
        raise RuntimeError("native load failed")

    monkeypatch.setattr(
        "rl_fzerox.ui.watch.runtime.live.worker.open_watch_runtime_session",
        _raise_bootstrap_error,
    )
    monkeypatch.setattr(
        "rl_fzerox.ui.watch.runtime.live.worker.publish_worker_message",
        lambda _queue, message: published.append(message),
    )

    run_simulation_worker(
        config=config,
        command_queue=object(),  # type: ignore[arg-type]
        snapshot_queue=object(),  # type: ignore[arg-type]
    )

    assert len(published) == 2
    error = published[0]
    assert isinstance(error, WorkerError)
    assert "watch worker failed during bootstrap: RuntimeError: native load failed" in error.message
    assert f"core_path={core_path.resolve()} file size=4 readable=yes" in error.message
    assert f"rom_path={rom_path.resolve()} file size=3 readable=yes" in error.message
    assert f"runtime_dir={runtime_dir.resolve()} missing parent_exists=yes" in error.message
    assert "renderer=gliden64" in error.message
    assert isinstance(published[1], WorkerClosed)
