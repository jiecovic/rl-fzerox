# src/rl_fzerox/ui/watch/runtime/career_mode/loop/entry.py
from __future__ import annotations

from multiprocessing.queues import Queue as ProcessQueue

from rl_fzerox.core.career_mode.controller import CareerModeController
from rl_fzerox.core.runtime_spec.schema import WatchAppConfig
from rl_fzerox.ui.watch.runtime.career_mode.loop.debug import (
    observe_career_mode_debug_trace,
    open_career_mode_debug_trace,
)
from rl_fzerox.ui.watch.runtime.career_mode.loop.runner import _run_career_mode_loop_body
from rl_fzerox.ui.watch.runtime.career_mode.loop.signals import CareerModeWorkerQuit
from rl_fzerox.ui.watch.runtime.career_mode.loop.state import (
    initial_career_mode_loop_state,
    publish_initial_career_snapshot,
)
from rl_fzerox.ui.watch.runtime.career_mode.recording import open_career_mode_recorder
from rl_fzerox.ui.watch.runtime.career_mode.session import (
    CareerModeRuntimeSession,
)


def run_loaded_career_mode_loop(
    *,
    config: WatchAppConfig,
    session: CareerModeRuntimeSession,
    controller: CareerModeController,
    command_queue: ProcessQueue,
    snapshot_queue: ProcessQueue,
) -> None:
    state = initial_career_mode_loop_state(
        config=config,
        session=session,
        controller=controller,
    )
    recorder = open_career_mode_recorder(
        config=config,
        native_fps=session.native_fps,
        native_sample_rate=session.native_sample_rate,
    )
    debug_trace = open_career_mode_debug_trace(config)

    try:
        publish_initial_career_snapshot(
            config=config,
            session=session,
            snapshot_queue=snapshot_queue,
            state=state,
            frame_recorder=recorder,
        )
        observe_career_mode_debug_trace(
            debug_trace,
            stage="initial",
            info=state.info,
            controller=controller,
            frame_source=session.render,
            force=True,
        )
        _run_career_mode_loop_body(
            config=config,
            session=session,
            controller=controller,
            command_queue=command_queue,
            snapshot_queue=snapshot_queue,
            state=state,
            frame_recorder=recorder,
            debug_trace=debug_trace,
        )
    except CareerModeWorkerQuit:
        return
    finally:
        if recorder is not None:
            recorder.close()
