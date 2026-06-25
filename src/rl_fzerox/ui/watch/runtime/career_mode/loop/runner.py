# src/rl_fzerox/ui/watch/runtime/career_mode/loop/runner.py
"""Career Mode watch-loop body.

This worker-side loop coordinates process commands, menu stepping, policy
handoff, snapshot publication, recording notices, and timing. The core Career
Mode controller remains the owner of FSM and attempt lifecycle decisions.
"""

from __future__ import annotations

import time
from multiprocessing.queues import Queue as ProcessQueue

from fzerox_emulator import SpinRequest
from rl_fzerox.core.career_mode.controller import CareerModeController
from rl_fzerox.core.career_mode.navigation import (
    MenuInput,
    RawMenuStep,
    course_id_from_info,
    in_gp_race,
)
from rl_fzerox.core.runtime_spec.schema import WatchAppConfig
from rl_fzerox.ui.watch.live_series import EpisodeLiveSeriesTracker
from rl_fzerox.ui.watch.records import TrackRecordBook
from rl_fzerox.ui.watch.runtime.career_mode.loop.commands import (
    apply_control_timing_command,
    apply_runtime_command_state,
)
from rl_fzerox.ui.watch.runtime.career_mode.loop.debug import (
    CareerModeDebugTrace,
    CareerModeLoopTracer,
)
from rl_fzerox.ui.watch.runtime.career_mode.loop.policy_runtime import (
    CareerPolicyRuntime,
)
from rl_fzerox.ui.watch.runtime.career_mode.loop.recording import (
    drain_recording_notices,
    handle_controller_lifecycle,
)
from rl_fzerox.ui.watch.runtime.career_mode.loop.runtime import (
    active_policy_intro_target_timer,
    career_mode_attempt_id,
    career_runtime_error_context,
    fresh_menu_runtime_state,
    policy_intro_wait_required,
    reset_emulator_for_next_attempt,
    should_observe_policy_transition,
)
from rl_fzerox.ui.watch.runtime.career_mode.loop.signals import CareerModeWorkerQuit
from rl_fzerox.ui.watch.runtime.career_mode.loop.snapshot import (
    publish_career_loop_snapshot,
)
from rl_fzerox.ui.watch.runtime.career_mode.loop.state import (
    CareerModeLoopState,
    TimedRecordingNotice,
)
from rl_fzerox.ui.watch.runtime.career_mode.menu import (
    menu_viewer_info,
    step_menu,
)
from rl_fzerox.ui.watch.runtime.career_mode.policy_step import (
    required_episode_return,
    step_policy_or_manual,
)
from rl_fzerox.ui.watch.runtime.career_mode.recording import FrameRecorder
from rl_fzerox.ui.watch.runtime.career_mode.save_ram import (
    CareerModeSaveBinding,
    persist_career_mode_save_ram,
)
from rl_fzerox.ui.watch.runtime.career_mode.session import CareerModeRuntimeSession
from rl_fzerox.ui.watch.runtime.career_mode.timing import active_policy_timing
from rl_fzerox.ui.watch.runtime.ipc import drain_worker_commands
from rl_fzerox.ui.watch.runtime.observation import (
    apply_watch_state_feature_zeroing,
    toggle_watch_state_feature,
)
from rl_fzerox.ui.watch.runtime.policy.runner import _reset_policy_runner
from rl_fzerox.ui.watch.runtime.policy.visualization import (
    current_auxiliary_predictions as _current_auxiliary_predictions,
)
from rl_fzerox.ui.watch.runtime.policy.visualization import (
    current_auxiliary_targets as _current_auxiliary_targets,
)
from rl_fzerox.ui.watch.runtime.policy.visualization import (
    refresh_paused_cnn_activations as _refresh_paused_cnn_activations,
)
from rl_fzerox.ui.watch.runtime.telemetry import _read_live_telemetry


def _run_career_mode_loop_body(
    *,
    config: WatchAppConfig,
    session: CareerModeRuntimeSession,
    controller: CareerModeController,
    save_binding: CareerModeSaveBinding,
    command_queue: ProcessQueue,
    snapshot_queue: ProcessQueue,
    state: CareerModeLoopState,
    frame_recorder: FrameRecorder | None = None,
    debug_trace: CareerModeDebugTrace | None = None,
) -> None:
    control_rate = state.control_rate
    native_control_fps = state.native_control_fps
    target_control_fps = state.target_control_fps
    target_control_seconds = state.target_control_seconds
    native_frame_seconds = state.native_frame_seconds
    next_step_time = state.next_step_time
    paused = state.paused
    deterministic_policy = state.deterministic_policy
    manual_control_state = state.manual_control_state
    episode = state.episode
    cnn_visualization_enabled = state.cnn_visualization_enabled
    auxiliary_visualization_enabled = state.auxiliary_visualization_enabled
    live_visualization_enabled = state.live_visualization_enabled
    live_series = state.live_series
    track_record_book = state.track_record_book
    last_live_series_publish_time = state.last_live_series_publish_time
    cnn_normalization = state.cnn_normalization
    cnn_sampler = state.cnn_sampler
    watch_zeroed_state_features = state.watch_zeroed_state_features
    auxiliary_target_names = state.auxiliary_target_names
    policy_runtime = CareerPolicyRuntime.from_loop_state(state)
    raw_observation = state.raw_observation
    observation = state.observation
    raw_info = state.raw_info
    info = state.info
    track_record_attempt_id = career_mode_attempt_id(info)
    reset_info = state.reset_info
    current_telemetry = state.current_telemetry
    last_menu_step = state.last_menu_step
    manual_spin_request: SpinRequest = "none"
    recording_notice = TimedRecordingNotice()
    tracer = CareerModeLoopTracer(
        trace=debug_trace,
        controller=controller,
        frame_source=session.render,
    )

    def sync_attempt_change_side_effects() -> None:
        nonlocal track_record_attempt_id, track_record_book

        current_attempt_id = career_mode_attempt_id(info)
        if current_attempt_id == track_record_attempt_id:
            return
        track_record_book = TrackRecordBook()
        track_record_attempt_id = current_attempt_id

    def publish_snapshot(*, policy_visible: bool) -> None:
        publish_career_loop_snapshot(
            config=config,
            session=session,
            controller=controller,
            snapshot_queue=snapshot_queue,
            control_rate=control_rate,
            native_control_fps=native_control_fps,
            target_control_fps=target_control_fps,
            policy_visible=policy_visible,
            active_policy_control=policy_runtime.policy_control,
            observation=observation,
            info=info,
            reset_info=reset_info,
            episode=episode,
            episode_reward=policy_runtime.episode_reward,
            current_control_state=policy_runtime.control_state,
            current_gas_level=policy_runtime.gas_level,
            boost_lamp_level=policy_runtime.boost_lamp_level,
            current_policy_action=policy_runtime.policy_action,
            current_auxiliary_predictions=policy_runtime.auxiliary_predictions,
            current_auxiliary_targets=policy_runtime.auxiliary_targets,
            auxiliary_visualization_enabled=auxiliary_visualization_enabled,
            auxiliary_target_names=auxiliary_target_names,
            deterministic_policy=deterministic_policy,
            manual_control_enabled=policy_runtime.manual_enabled,
            cnn_activations=policy_runtime.cnn_activations,
            track_record_book=track_record_book,
            recording_notice=recording_notice,
            now=time.perf_counter(),
        )

    def reset_for_next_attempt_snapshot() -> None:
        nonlocal raw_info, info, current_telemetry, reset_info

        tracer.state(
            "before_emulator_reset",
            info=info,
            event="reset_requested",
            force=True,
        )
        raw_info, info, current_telemetry = reset_emulator_for_next_attempt(
            config=config,
            session=session,
            controller=controller,
            save_binding=save_binding,
        )
        reset_info = dict(info)
        policy_runtime.clear()
        sync_attempt_change_side_effects()
        tracer.state(
            "after_emulator_reset",
            info=info,
            event="reset_complete",
            force=True,
        )
        publish_snapshot(policy_visible=False)

    def refresh_after_terminal_episode(*, increment_episode: bool) -> None:
        nonlocal track_record_book, raw_observation, observation
        nonlocal live_series, last_live_series_publish_time, raw_info, info
        nonlocal current_telemetry, reset_info, episode

        track_record_book = track_record_book.update(
            info,
            current_telemetry,
            episode_done=True,
        )
        control_rate.reset()
        if increment_episode:
            episode += 1
        raw_observation = None
        observation = None
        live_series = EpisodeLiveSeriesTracker()
        last_live_series_publish_time = 0.0
        raw_info, info, current_telemetry = fresh_menu_runtime_state(session)
        info = controller.viewer_info(
            info=info,
            active_policy_control=None,
        )
        reset_info = dict(info)
        sync_attempt_change_side_effects()
        policy_runtime.clear(reset_episode_reward=True)
        tracer.state("after_terminal_refresh", info=info)
        publish_snapshot(policy_visible=False)

    def run_menu_step(step: RawMenuStep) -> None:
        nonlocal last_menu_step, current_step_seconds, raw_observation, observation

        last_menu_step = step
        step_menu(
            config=config,
            session=session,
            controller=controller,
            snapshot_queue=snapshot_queue,
            step=step,
            info=info,
            reset_info=reset_info,
            episode=episode,
            episode_reward=policy_runtime.episode_reward,
            control_rate=control_rate,
            target_control_fps=target_control_fps,
            native_frame_seconds=native_frame_seconds,
            deterministic_policy=deterministic_policy,
            track_record_book=track_record_book,
            frame_recorder=frame_recorder,
        )
        current_step_seconds = None
        raw_observation = None
        observation = None

    def refresh_menu_viewer_state(
        *,
        reset_controls: bool = False,
        refresh_telemetry: bool = True,
    ) -> None:
        nonlocal raw_info, info, reset_info, current_telemetry

        if reset_controls:
            policy_runtime.reset_menu_controls()
        raw_info = menu_viewer_info(session)
        info = controller.viewer_info(
            info=dict(raw_info),
            active_policy_control=None,
        )
        reset_info = dict(info)
        if refresh_telemetry:
            current_telemetry = _read_live_telemetry(session.emulator)
        sync_attempt_change_side_effects()

    def step_idle_after_runner_complete() -> None:
        policy_runtime.clear(reset_episode_reward=True)
        run_menu_step(
            RawMenuStep(
                menu_input=MenuInput.NEUTRAL,
                frames=1,
                phase="career_complete:idle",
            )
        )
        refresh_menu_viewer_state()

    def step_visible_policy_intro_wait(target_timer: int) -> None:
        run_menu_step(
            RawMenuStep(
                menu_input=MenuInput.NEUTRAL,
                frames=1,
                phase=f"policy_intro:wait_until_timer_{target_timer}",
            )
        )
        refresh_menu_viewer_state(reset_controls=True)

    def begin_active_policy_race() -> None:
        nonlocal raw_observation, raw_info, watch_zeroed_state_features
        nonlocal auxiliary_target_names, live_series, last_live_series_publish_time
        nonlocal observation, info, reset_info, current_telemetry

        policy_control = policy_runtime.policy_control
        if policy_control is None:
            raise RuntimeError("Career Mode policy control was not resolved.")
        control_rate.reset()
        raw_observation, raw_info = session.begin_policy_race(
            policy_control=policy_control,
            seed=config.watch.attempt_seed,
            course_id=course_id_from_info(info),
        )
        watch_zeroed_state_features = session.watch_zeroed_state_features
        auxiliary_target_names = session.auxiliary_target_names
        live_series = EpisodeLiveSeriesTracker()
        last_live_series_publish_time = 0.0
        observation, info = apply_watch_state_feature_zeroing(
            raw_observation,
            raw_info,
            watch_zeroed_features=watch_zeroed_state_features,
        )
        info = controller.viewer_info(
            info=info,
            active_policy_control=policy_control,
        )
        reset_info = dict(info)
        sync_attempt_change_side_effects()
        policy_runtime.mark_policy_race_started(
            episode_reward=required_episode_return(info),
            control_state=session.last_requested_control_state,
            gas_level=session.last_gas_level,
        )
        current_telemetry = _read_live_telemetry(session.emulator)
        _reset_policy_runner(policy_control.runner)
        tracer.state("begin_policy_race", info=info, force=True)
        publish_snapshot(policy_visible=True)

    try:
        while True:
            previous_cnn_visualization_enabled = cnn_visualization_enabled
            previous_auxiliary_visualization_enabled = auxiliary_visualization_enabled
            previous_live_visualization_enabled = live_visualization_enabled
            previous_cnn_normalization = cnn_normalization
            if recording_notices := drain_recording_notices(frame_recorder):
                recording_notice.show(recording_notices[-1], now=time.perf_counter())
                publish_snapshot(policy_visible=controller.policy_owns_control())
            commands, paused, manual_control_state = drain_worker_commands(
                command_queue,
                paused=paused,
                control_state=manual_control_state,
                spin_request=manual_spin_request,
                manual_control_enabled=policy_runtime.manual_enabled,
                cnn_visualization_enabled=cnn_visualization_enabled,
                auxiliary_visualization_enabled=auxiliary_visualization_enabled,
                live_visualization_enabled=live_visualization_enabled,
                cnn_normalization=cnn_normalization,
            )
            if commands.quit_requested:
                raise CareerModeWorkerQuit()
            if commands.save_requests:
                persist_career_mode_save_ram(save_binding, session)
            timing_update = apply_control_timing_command(
                commands=commands,
                session=session,
                control_rate=control_rate,
                native_control_fps=native_control_fps,
                target_control_fps=target_control_fps,
            )
            if timing_update is not None:
                target_control_fps = timing_update.target_control_fps
                target_control_seconds = timing_update.target_control_seconds
                native_frame_seconds = timing_update.native_frame_seconds
                next_step_time = timing_update.next_step_time
            current_step_seconds = target_control_seconds
            if commands.toggle_zeroed_state_feature_name is not None:
                watch_zeroed_state_features = toggle_watch_state_feature(
                    watch_zeroed_state_features,
                    commands.toggle_zeroed_state_feature_name,
                )
                if raw_observation is not None:
                    observation, info = apply_watch_state_feature_zeroing(
                        raw_observation,
                        raw_info,
                        watch_zeroed_features=watch_zeroed_state_features,
                    )
                    info = controller.viewer_info(
                        info=info,
                        active_policy_control=policy_runtime.policy_control,
                    )
                else:
                    observation = None
                    info = controller.viewer_info(
                        info=dict(raw_info),
                        active_policy_control=None,
                    )
                publish_snapshot(policy_visible=controller.policy_owns_control())

            command_state = apply_runtime_command_state(
                commands=commands,
                auxiliary_target_names=auxiliary_target_names,
                previous_live_visualization_enabled=previous_live_visualization_enabled,
                last_live_series_publish_time=last_live_series_publish_time,
                deterministic_policy=deterministic_policy,
                policy_owns_control=controller.policy_owns_control(),
                active_policy_started=policy_runtime.started,
            )
            cnn_visualization_enabled = command_state.cnn_visualization_enabled
            auxiliary_visualization_enabled = command_state.auxiliary_visualization_enabled
            live_visualization_enabled = command_state.live_visualization_enabled
            last_live_series_publish_time = command_state.last_live_series_publish_time
            cnn_normalization = command_state.cnn_normalization
            deterministic_policy = command_state.deterministic_policy
            next_manual_control_enabled = command_state.manual_control_enabled
            manual_spin_request = command_state.spin_request
            if (
                auxiliary_visualization_enabled != previous_auxiliary_visualization_enabled
                and controller.policy_owns_control()
                and observation is not None
            ):
                policy_runtime.auxiliary_predictions = _current_auxiliary_predictions(
                    policy_runner=(
                        policy_runtime.policy_control.runner
                        if policy_runtime.policy_control is not None
                        else None
                    ),
                    enabled=auxiliary_visualization_enabled,
                    observation=observation,
                    target_names=auxiliary_target_names,
                )
                policy_runtime.auxiliary_targets = _current_auxiliary_targets(
                    telemetry=current_telemetry,
                    enabled=auxiliary_visualization_enabled,
                    target_names=auxiliary_target_names,
                )

            if paused and commands.step_requests <= 0:
                if controller.policy_owns_control() and observation is not None:
                    (
                        policy_runtime.cnn_activations,
                        cnn_snapshot_changed,
                    ) = _refresh_paused_cnn_activations(
                        current_activations=policy_runtime.cnn_activations,
                        cnn_sampler=cnn_sampler,
                        cnn_visualization_enabled=cnn_visualization_enabled,
                        previous_cnn_visualization_enabled=previous_cnn_visualization_enabled,
                        cnn_normalization=cnn_normalization,
                        previous_cnn_normalization=previous_cnn_normalization,
                        policy_runner=(
                            policy_runtime.policy_control.runner
                            if policy_runtime.policy_control is not None
                            else None
                        ),
                        observation=observation,
                    )
                    if cnn_snapshot_changed:
                        publish_snapshot(policy_visible=True)
                time.sleep(0.01)
                continue
            if not paused and target_control_seconds is not None:
                wait_seconds = next_step_time - time.perf_counter()
                if wait_seconds > 0.0:
                    time.sleep(min(wait_seconds, 0.005))
                    continue

            before_step_handled = controller.before_step(session=session, info=info)
            if before_step_handled:
                raw_info, info, current_telemetry = fresh_menu_runtime_state(session)
                info = controller.viewer_info(
                    info=info,
                    active_policy_control=policy_runtime.policy_control,
                )
                sync_attempt_change_side_effects()
                tracer.state("after_before_step", info=info)
            lifecycle = handle_controller_lifecycle(
                controller=controller,
                frame_recorder=frame_recorder,
                info=info,
            )
            tracer.lifecycle(lifecycle, info=info)
            if lifecycle.reset_requested:
                reset_for_next_attempt_snapshot()
                continue
            if before_step_handled:
                reset_info = dict(info)
                if not lifecycle.has_active_attempt:
                    step_idle_after_runner_complete()
                    continue
            policy_owns_control = controller.policy_owns_control()
            if should_observe_policy_transition(
                policy_owns_control=policy_owns_control,
                active_policy_started=policy_runtime.started,
                info=info,
            ):
                terminal_handled = controller.observe_step(
                    session=session,
                    info=info,
                )
                if not terminal_handled and not in_gp_race(info):
                    raise RuntimeError("Career Mode left a race before observing a game result")
                if terminal_handled:
                    tracer.state(
                        "terminal_handled",
                        info=info,
                        event="policy_start_terminal",
                        force=True,
                    )
                    terminal_info = controller.viewer_info(
                        info=info,
                        active_policy_control=policy_runtime.policy_control,
                    )
                    lifecycle = handle_controller_lifecycle(
                        controller=controller,
                        frame_recorder=frame_recorder,
                        info=terminal_info,
                        record_event=True,
                    )
                    tracer.lifecycle(lifecycle, info=info, trace_info=terminal_info)
                    if lifecycle.reset_requested:
                        reset_for_next_attempt_snapshot()
                        continue
                    refresh_after_terminal_episode(increment_episode=False)
                    if not lifecycle.has_active_attempt:
                        step_idle_after_runner_complete()
                        continue
            if not controller.has_active_attempt():
                step_idle_after_runner_complete()
                continue
            menu_step = controller.next_raw_step(info=info)
            if menu_step is not None:
                policy_runtime.clear(reset_episode_reward=True)
                run_menu_step(menu_step)
                refresh_menu_viewer_state()
                tracer.state("after_menu_step", info=info)
            else:
                policy_runtime.policy_control = controller.active_policy_control(
                    session=session,
                    current_policy_control=policy_runtime.policy_control,
                    info=info,
                )
                policy_control = policy_runtime.policy_control
                if policy_control is None:
                    step = RawMenuStep(
                        menu_input=MenuInput.NEUTRAL,
                        frames=1,
                        phase="policy_resolution:wait",
                    )
                    run_menu_step(step)
                    refresh_menu_viewer_state(
                        reset_controls=True,
                        refresh_telemetry=False,
                    )
                    tracer.state("after_policy_resolution_wait", info=info)
                    continue
                if not policy_runtime.started:
                    target_timer = active_policy_intro_target_timer(policy_control)
                    if target_timer is not None and policy_intro_wait_required(
                        info=info,
                        target_timer=target_timer,
                    ):
                        step_visible_policy_intro_wait(target_timer)
                        tracer.state("after_policy_intro_wait", info=info)
                        continue
                    begin_active_policy_race()
                    continue

                if observation is None:
                    raise RuntimeError(
                        "Career Mode policy control started without a policy observation."
                    )
                policy_timing = active_policy_timing(
                    config,
                    session,
                    native_control_fps=native_control_fps,
                    target_control_fps=target_control_fps,
                )
                current_step_seconds = policy_timing.target_seconds
                policy_runtime.manual_enabled = next_manual_control_enabled
                if policy_runtime.manual_enabled:
                    policy_runtime.control_state = commands.control_state
                policy_step = step_policy_or_manual(
                    config=config,
                    session=session,
                    controller=controller,
                    snapshot_queue=snapshot_queue,
                    active_policy_control=policy_control,
                    policy_runner=policy_control.runner,
                    observation=observation,
                    info=info,
                    reset_info=reset_info,
                    episode=episode,
                    episode_reward=policy_runtime.episode_reward,
                    control_rate=control_rate,
                    target_policy_control_fps=policy_timing.target_fps,
                    target_control_seconds=policy_timing.target_seconds,
                    deterministic_policy=deterministic_policy,
                    frame_recorder=frame_recorder,
                    manual_control_enabled=policy_runtime.manual_enabled,
                    current_control_state=policy_runtime.control_state,
                    spin_request=(manual_spin_request if policy_runtime.manual_enabled else "none"),
                    boost_lamp_level=policy_runtime.boost_lamp_level,
                    cnn_visualization_enabled=cnn_visualization_enabled,
                    cnn_normalization=cnn_normalization,
                    cnn_sampler=cnn_sampler,
                    auxiliary_visualization_enabled=auxiliary_visualization_enabled,
                    auxiliary_target_names=auxiliary_target_names,
                    watch_zeroed_state_features=watch_zeroed_state_features,
                    live_visualization_enabled=live_visualization_enabled,
                    live_series=live_series,
                    last_live_series_publish_time=last_live_series_publish_time,
                    track_record_book=track_record_book,
                )
                raw_observation = policy_step.raw_observation
                raw_info = policy_step.raw_info
                observation = policy_step.observation
                info = policy_step.info
                policy_runtime.apply_policy_step(policy_step)
                current_telemetry = policy_step.telemetry
                last_live_series_publish_time = policy_step.last_live_series_publish_time
                tracer.state("policy_step", info=info)
                terminal_handled = controller.observe_step(
                    session=session,
                    info=info,
                )
                if terminal_handled and not controller.policy_owns_control():
                    tracer.state(
                        "terminal_handled",
                        info=info,
                        event="race_terminal",
                        force=True,
                    )
                    terminal_info = controller.viewer_info(
                        info=info,
                        active_policy_control=policy_runtime.policy_control,
                    )
                    lifecycle = handle_controller_lifecycle(
                        controller=controller,
                        frame_recorder=frame_recorder,
                        info=terminal_info,
                        record_event=True,
                    )
                    tracer.lifecycle(lifecycle, info=info, trace_info=terminal_info)
                    if lifecycle.reset_requested:
                        reset_for_next_attempt_snapshot()
                        continue
                    refresh_after_terminal_episode(increment_episode=True)
                    if not lifecycle.has_active_attempt:
                        step_idle_after_runner_complete()
                        continue

            if current_step_seconds is not None:
                now = time.perf_counter()
                next_step_time = max(next_step_time + current_step_seconds, now)
    except CareerModeWorkerQuit:
        raise
    except Exception as exc:
        tracer.state(
            "worker_exception",
            info=info,
            event=type(exc).__name__,
            force=True,
        )
        raise RuntimeError(
            career_runtime_error_context(
                exc,
                controller=controller,
                info=info,
                last_menu_step=last_menu_step,
            )
        ) from exc
