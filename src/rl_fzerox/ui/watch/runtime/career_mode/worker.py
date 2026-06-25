# src/rl_fzerox/ui/watch/runtime/career_mode/worker.py
from __future__ import annotations

from multiprocessing.queues import Queue as ProcessQueue

from rl_fzerox.core.career_mode.controller import CareerModeController
from rl_fzerox.core.runtime_spec.schema import WatchAppConfig
from rl_fzerox.ui.watch.runtime.career_mode.attempts import (
    RUNNER_CLOSE_REASONS,
    fail_running_attempts,
)
from rl_fzerox.ui.watch.runtime.career_mode.loop import run_loaded_career_mode_loop
from rl_fzerox.ui.watch.runtime.career_mode.loop.runtime import (
    should_observe_policy_transition,
)
from rl_fzerox.ui.watch.runtime.career_mode.save_ram import (
    CareerModeSaveBinding,
    career_mode_save_binding_from_config,
    load_career_mode_save_ram,
    persist_career_mode_save_ram,
)
from rl_fzerox.ui.watch.runtime.career_mode.session import (
    CareerModeRuntimeSession,
    open_career_mode_runtime_session,
)
from rl_fzerox.ui.watch.runtime.ipc import (
    WorkerClosed,
    WorkerError,
    publish_worker_message,
)


def run_career_mode_worker(
    config: WatchAppConfig,
    command_queue: ProcessQueue,
    snapshot_queue: ProcessQueue,
) -> None:
    """Run Career Mode from the portable save file and menu FSM."""

    try:
        _run_career_mode_loop(config, command_queue, snapshot_queue)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        _mark_runner_failed(config)
        publish_worker_message(snapshot_queue, WorkerError(message=str(exc)))
    finally:
        publish_worker_message(snapshot_queue, WorkerClosed())


def _run_career_mode_loop(
    config: WatchAppConfig,
    command_queue: ProcessQueue,
    snapshot_queue: ProcessQueue,
) -> None:
    session = open_career_mode_runtime_session(config)
    controller = CareerModeController.from_config(config)
    save_binding = career_mode_save_binding_from_config(config)
    load_career_mode_save_ram(save_binding, session)

    failure_reason = RUNNER_CLOSE_REASONS.closed
    try:
        run_loaded_career_mode_loop(
            config=config,
            session=session,
            controller=controller,
            save_binding=save_binding,
            command_queue=command_queue,
            snapshot_queue=snapshot_queue,
        )
    except BaseException:
        failure_reason = RUNNER_CLOSE_REASONS.failed
        raise
    finally:
        _close_career_mode(
            session,
            save_binding,
            failure_reason=failure_reason,
        )
        session.close()


def _close_career_mode(
    session: CareerModeRuntimeSession,
    save_binding: CareerModeSaveBinding,
    *,
    failure_reason: str,
) -> None:
    persist_career_mode_save_ram(save_binding, session)
    fail_running_attempts(
        save_binding.store,
        save_game_id=save_binding.save_game_id,
        failure_reason=failure_reason,
    )


def _mark_runner_failed(config: WatchAppConfig) -> None:
    save_game_id = config.watch.managed_save_game_id
    if save_game_id is not None:
        save_binding = career_mode_save_binding_from_config(config)
        fail_running_attempts(
            save_binding.store,
            save_game_id=save_binding.save_game_id,
            failure_reason=RUNNER_CLOSE_REASONS.failed,
        )


def _should_observe_policy_transition(
    *,
    policy_owns_control: bool,
    active_policy_started: bool,
    info: dict[str, object],
) -> bool:
    return should_observe_policy_transition(
        policy_owns_control=policy_owns_control,
        active_policy_started=active_policy_started,
        info=info,
    )
