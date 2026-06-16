# src/rl_fzerox/ui/watch/runtime/career_mode/loop/runner.py
from __future__ import annotations

import time
from multiprocessing.queues import Queue as ProcessQueue

from fzerox_emulator import FZeroXTelemetry, RaceControlState, SpinRequest
from rl_fzerox.core.career_mode.runner.controller import CareerModeController
from rl_fzerox.core.career_mode.runner.menu import (
    MenuInput,
    RawMenuStep,
    course_id_from_info,
    in_gp_race,
)
from rl_fzerox.core.career_mode.runner.save_file import (
    persist_save_ram,
)
from rl_fzerox.core.runtime_spec.schema import WatchAppConfig
from rl_fzerox.ui.watch.live_series import EpisodeLiveSeriesTracker
from rl_fzerox.ui.watch.runtime.career_mode.loop.recording import (
    drain_recording_notices,
    finish_pending_recording_segment,
    record_controller_event,
)
from rl_fzerox.ui.watch.runtime.career_mode.loop.state import (
    CareerModeLoopState,
    TimedRecordingNotice,
    initial_career_mode_loop_state,
    publish_initial_career_snapshot,
)
from rl_fzerox.ui.watch.runtime.career_mode.menu import (
    menu_viewer_info,
    reset_race_progress_info,
    step_menu,
)
from rl_fzerox.ui.watch.runtime.career_mode.policy_step import (
    required_episode_return,
    step_policy_or_manual,
)
from rl_fzerox.ui.watch.runtime.career_mode.recording import (
    FrameRecorder,
    open_career_mode_recorder,
)
from rl_fzerox.ui.watch.runtime.career_mode.session import (
    CareerModeRuntimeSession,
)
from rl_fzerox.ui.watch.runtime.career_mode.timing import (
    active_policy_timing,
    measured_game_fps,
    set_session_control_timing,
    target_game_fps,
    with_measured_game_fps,
)
from rl_fzerox.ui.watch.runtime.career_mode.timing import (
    native_frame_seconds as _native_frame_seconds,
)
from rl_fzerox.ui.watch.runtime.career_mode.timing import (
    snapshot_action_repeat as _snapshot_action_repeat,
)
from rl_fzerox.ui.watch.runtime.career_mode.timing import (
    snapshot_target_control_fps as _snapshot_target_control_fps,
)
from rl_fzerox.ui.watch.runtime.ipc import (
    drain_worker_commands,
    publish_worker_message,
)
from rl_fzerox.ui.watch.runtime.observation import (
    apply_watch_state_feature_zeroing,
    toggle_watch_state_feature,
)
from rl_fzerox.ui.watch.runtime.policy import (
    _policy_reload_error,
    _reset_policy_runner,
)
from rl_fzerox.ui.watch.runtime.snapshots import _build_snapshot
from rl_fzerox.ui.watch.runtime.telemetry import _read_live_telemetry
from rl_fzerox.ui.watch.runtime.timing import (
    _adjust_control_fps,
    _target_seconds,
)
from rl_fzerox.ui.watch.runtime.visualization import (
    current_auxiliary_predictions as _current_auxiliary_predictions,
)
from rl_fzerox.ui.watch.runtime.visualization import (
    current_auxiliary_targets as _current_auxiliary_targets,
)
from rl_fzerox.ui.watch.runtime.visualization import (
    refresh_paused_cnn_activations as _refresh_paused_cnn_activations,
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

    try:
        publish_initial_career_snapshot(
            config=config,
            session=session,
            snapshot_queue=snapshot_queue,
            state=state,
            frame_recorder=recorder,
        )
        _run_career_mode_loop_body(
            config=config,
            session=session,
            controller=controller,
            command_queue=command_queue,
            snapshot_queue=snapshot_queue,
            state=state,
            frame_recorder=recorder,
        )
    except _CareerModeWorkerQuit:
        return
    finally:
        if recorder is not None:
            recorder.close()


class _CareerModeWorkerQuit(Exception):
    """Internal signal used to unwind the Career Mode worker loop."""


def _run_career_mode_loop_body(
    *,
    config: WatchAppConfig,
    session: CareerModeRuntimeSession,
    controller: CareerModeController,
    command_queue: ProcessQueue,
    snapshot_queue: ProcessQueue,
    state: CareerModeLoopState,
    frame_recorder: FrameRecorder | None = None,
) -> None:
    control_rate = state.control_rate
    native_control_fps = state.native_control_fps
    target_control_fps = state.target_control_fps
    target_control_seconds = state.target_control_seconds
    native_frame_seconds = state.native_frame_seconds
    next_step_time = state.next_step_time
    paused = state.paused
    deterministic_policy = state.deterministic_policy
    manual_control_enabled = state.manual_control_enabled
    manual_control_state = state.manual_control_state
    current_control_state = state.current_control_state
    current_gas_level = state.current_gas_level
    boost_lamp_level = state.boost_lamp_level
    episode = state.episode
    episode_reward = state.episode_reward
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
    active_policy_control = state.active_policy_control
    active_policy_started = state.active_policy_started
    current_policy_action = state.current_policy_action
    raw_observation = state.raw_observation
    observation = state.observation
    raw_info = state.raw_info
    info = state.info
    reset_info = state.reset_info
    current_telemetry = state.current_telemetry
    current_auxiliary_predictions = state.current_auxiliary_predictions
    current_auxiliary_targets = state.current_auxiliary_targets
    cnn_activations = state.cnn_activations
    last_menu_step = state.last_menu_step
    manual_spin_request: SpinRequest = "none"
    recording_notice = TimedRecordingNotice()

    def publish_snapshot(*, policy_visible: bool) -> None:
        policy_active = policy_visible and observation is not None
        snapshot_target_fps = _snapshot_target_control_fps(
            config=config,
            session=session,
            native_control_fps=native_control_fps,
            target_control_fps=target_control_fps,
            policy_active=policy_active,
        )
        snapshot_info = controller.viewer_info(
            info=info if policy_active else reset_race_progress_info(info),
            active_policy_control=(active_policy_control if policy_active else None),
        )
        snapshot_config = session.snapshot_config(config) if policy_active else config
        snapshot_repeat = _snapshot_action_repeat(
            snapshot_config,
            policy_active=policy_active,
        )
        snapshot_info = with_measured_game_fps(
            snapshot_info,
            game_fps=measured_game_fps(
                control_fps=control_rate.rate_hz(),
                action_repeat=snapshot_repeat,
            ),
            game_fps_target=target_game_fps(
                target_control_fps=snapshot_target_fps,
                action_repeat=snapshot_repeat,
            ),
        )
        snapshot_info = recording_notice.apply(snapshot_info, now=time.perf_counter())
        runner = (
            active_policy_control.runner
            if policy_active and active_policy_control is not None
            else None
        )
        publish_worker_message(
            snapshot_queue,
            _build_snapshot(
                config=snapshot_config,
                env=session,
                emulator=session.emulator,
                observation=observation if policy_active else None,
                info=snapshot_info,
                reset_info=reset_info,
                episode=episode,
                episode_reward=episode_reward if policy_active else 0.0,
                control_fps=control_rate.rate_hz(),
                target_control_fps=snapshot_target_fps,
                action_repeat=snapshot_repeat,
                control_state=current_control_state,
                gas_level=current_gas_level,
                boost_lamp_level=boost_lamp_level,
                action_mask_branches=session.action_mask_branches(),
                policy_action=current_policy_action if policy_active else None,
                policy_runner=runner,
                policy_auxiliary_state_predictions=(
                    current_auxiliary_predictions if policy_active else None
                ),
                policy_auxiliary_state_targets=(
                    current_auxiliary_targets if policy_active else None
                ),
                include_auxiliary_state=(policy_active and auxiliary_visualization_enabled),
                auxiliary_target_names=auxiliary_target_names,
                deterministic_policy=deterministic_policy,
                manual_control_enabled=manual_control_enabled if policy_active else False,
                policy_reload_error=_policy_reload_error(runner),
                cnn_activations=cnn_activations if policy_active else None,
                track_record_book=track_record_book,
            ),
        )

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
                manual_control_enabled=manual_control_enabled,
                cnn_visualization_enabled=cnn_visualization_enabled,
                auxiliary_visualization_enabled=auxiliary_visualization_enabled,
                live_visualization_enabled=live_visualization_enabled,
                cnn_normalization=cnn_normalization,
            )
            if commands.quit_requested:
                raise _CareerModeWorkerQuit()
            if commands.save_requests:
                persist_save_ram(config, session)
            if commands.reset_control_fps:
                target_control_fps = native_control_fps
                target_control_seconds = _target_seconds(target_control_fps)
                native_frame_seconds = _native_frame_seconds(target_control_seconds)
                set_session_control_timing(
                    session,
                    target_control_fps=target_control_fps,
                    target_control_seconds=target_control_seconds,
                )
                control_rate.reset()
                next_step_time = time.perf_counter()
            elif commands.control_fps_delta:
                target_control_fps = _adjust_control_fps(
                    target_control_fps,
                    commands.control_fps_delta,
                    native_control_fps=native_control_fps,
                )
                target_control_seconds = _target_seconds(target_control_fps)
                native_frame_seconds = _native_frame_seconds(target_control_seconds)
                set_session_control_timing(
                    session,
                    target_control_fps=target_control_fps,
                    target_control_seconds=target_control_seconds,
                )
                control_rate.reset()
                next_step_time = time.perf_counter()
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
                        active_policy_control=active_policy_control,
                    )
                else:
                    observation = None
                    info = controller.viewer_info(
                        info=dict(raw_info),
                        active_policy_control=None,
                    )
                publish_snapshot(policy_visible=controller.policy_owns_control())

            cnn_visualization_enabled = commands.cnn_visualization_enabled
            auxiliary_visualization_enabled = (
                bool(auxiliary_target_names) and commands.auxiliary_visualization_enabled
            )
            live_visualization_enabled = commands.live_visualization_enabled
            if live_visualization_enabled != previous_live_visualization_enabled:
                last_live_series_publish_time = 0.0
            cnn_normalization = commands.cnn_normalization
            if commands.toggle_deterministic_policy:
                deterministic_policy = not deterministic_policy
            next_manual_control_enabled = (
                commands.manual_control_enabled
                if controller.policy_owns_control() and active_policy_started
                else False
            )
            if next_manual_control_enabled:
                manual_spin_request = commands.spin_request
            else:
                manual_spin_request = "none"
            if (
                auxiliary_visualization_enabled != previous_auxiliary_visualization_enabled
                and controller.policy_owns_control()
                and observation is not None
            ):
                current_auxiliary_predictions = _current_auxiliary_predictions(
                    policy_runner=(
                        active_policy_control.runner if active_policy_control is not None else None
                    ),
                    enabled=auxiliary_visualization_enabled,
                    observation=observation,
                    target_names=auxiliary_target_names,
                )
                current_auxiliary_targets = _current_auxiliary_targets(
                    telemetry=current_telemetry,
                    enabled=auxiliary_visualization_enabled,
                    target_names=auxiliary_target_names,
                )

            if paused and commands.step_requests <= 0:
                if controller.policy_owns_control() and observation is not None:
                    cnn_activations, cnn_snapshot_changed = _refresh_paused_cnn_activations(
                        current_activations=cnn_activations,
                        cnn_sampler=cnn_sampler,
                        cnn_visualization_enabled=cnn_visualization_enabled,
                        previous_cnn_visualization_enabled=previous_cnn_visualization_enabled,
                        cnn_normalization=cnn_normalization,
                        previous_cnn_normalization=previous_cnn_normalization,
                        policy_runner=(
                            active_policy_control.runner
                            if active_policy_control is not None
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
                raw_info, info, current_telemetry = _fresh_menu_runtime_state(session)
                info = controller.viewer_info(
                    info=info,
                    active_policy_control=active_policy_control,
                )
            finish_pending_recording_segment(
                controller=controller,
                frame_recorder=frame_recorder,
                info=info,
            )
            if before_step_handled:
                reset_info = dict(info)
                if not controller.has_active_attempt():
                    publish_snapshot(policy_visible=False)
                    raise _CareerModeWorkerQuit()
            policy_owns_control = controller.policy_owns_control()
            if _should_observe_policy_transition(
                policy_owns_control=policy_owns_control,
                active_policy_started=active_policy_started,
                info=info,
            ):
                terminal_handled = controller.observe_step(
                    session=session,
                    info=info,
                )
                if not terminal_handled and not in_gp_race(info):
                    raise RuntimeError("Career Mode left a race before observing a game result")
                if terminal_handled:
                    terminal_info = controller.viewer_info(
                        info=info,
                        active_policy_control=active_policy_control,
                    )
                    record_controller_event(
                        controller=controller,
                        frame_recorder=frame_recorder,
                        info=terminal_info,
                    )
                    track_record_book = track_record_book.update(
                        info,
                        current_telemetry,
                        episode_done=True,
                    )
                    control_rate.reset()
                    raw_observation = None
                    observation = None
                    episode_reward = 0.0
                    live_series = EpisodeLiveSeriesTracker()
                    last_live_series_publish_time = 0.0
                    raw_info, info, current_telemetry = _fresh_menu_runtime_state(session)
                    info = controller.viewer_info(
                        info=info,
                        active_policy_control=None,
                    )
                    reset_info = dict(info)
                    active_policy_control = None
                    active_policy_started = False
                    current_policy_action = None
                    current_control_state = RaceControlState()
                    current_gas_level = 0.0
                    boost_lamp_level = 0.0
                    cnn_activations = None
                    current_auxiliary_predictions = None
                    current_auxiliary_targets = None
                    manual_control_enabled = False
                    publish_snapshot(policy_visible=False)
                    if not controller.has_active_attempt():
                        raise _CareerModeWorkerQuit()
            menu_step = controller.next_raw_step(info=info)
            if menu_step is not None:
                last_menu_step = menu_step
                active_policy_control = None
                active_policy_started = False
                manual_control_enabled = False
                current_policy_action = None
                current_control_state = RaceControlState()
                current_gas_level = 0.0
                boost_lamp_level = 0.0
                episode_reward = 0.0
                cnn_activations = None
                step_menu(
                    config=config,
                    session=session,
                    controller=controller,
                    snapshot_queue=snapshot_queue,
                    step=menu_step,
                    info=info,
                    reset_info=reset_info,
                    episode=episode,
                    episode_reward=episode_reward,
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
                raw_info = menu_viewer_info(session)
                info = controller.viewer_info(
                    info=dict(raw_info),
                    active_policy_control=None,
                )
                reset_info = dict(info)
                current_telemetry = _read_live_telemetry(session.emulator)
            else:
                active_policy_control = controller.active_policy_control(
                    session=session,
                    current_policy_control=active_policy_control,
                    info=info,
                )
                if active_policy_control is None:
                    step = RawMenuStep(
                        menu_input=MenuInput.NEUTRAL,
                        frames=1,
                        phase="policy_resolution:wait",
                    )
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
                        episode_reward=episode_reward,
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
                    episode_reward = 0.0
                    current_control_state = RaceControlState()
                    current_gas_level = 0.0
                    boost_lamp_level = 0.0
                    raw_info = menu_viewer_info(session)
                    info = controller.viewer_info(
                        info=dict(raw_info),
                        active_policy_control=None,
                    )
                    reset_info = dict(info)
                    continue
                if not active_policy_started:
                    control_rate.reset()
                    raw_observation, raw_info = session.begin_policy_race(
                        policy_control=active_policy_control,
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
                        active_policy_control=active_policy_control,
                    )
                    reset_info = dict(info)
                    episode_reward = required_episode_return(info)
                    current_policy_action = None
                    current_control_state = session.last_requested_control_state
                    current_gas_level = session.last_gas_level
                    current_telemetry = _read_live_telemetry(session.emulator)
                    current_auxiliary_predictions = None
                    current_auxiliary_targets = None
                    _reset_policy_runner(active_policy_control.runner)
                    active_policy_started = True
                    publish_snapshot(policy_visible=True)
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
                manual_control_enabled = next_manual_control_enabled
                if manual_control_enabled:
                    current_control_state = commands.control_state
                policy_step = step_policy_or_manual(
                    config=config,
                    session=session,
                    controller=controller,
                    snapshot_queue=snapshot_queue,
                    active_policy_control=active_policy_control,
                    policy_runner=active_policy_control.runner,
                    observation=observation,
                    info=info,
                    reset_info=reset_info,
                    episode=episode,
                    episode_reward=episode_reward,
                    control_rate=control_rate,
                    target_policy_control_fps=policy_timing.target_fps,
                    target_control_seconds=policy_timing.target_seconds,
                    deterministic_policy=deterministic_policy,
                    frame_recorder=frame_recorder,
                    manual_control_enabled=manual_control_enabled,
                    current_control_state=current_control_state,
                    spin_request=manual_spin_request if manual_control_enabled else "none",
                    boost_lamp_level=boost_lamp_level,
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
                episode_reward = policy_step.episode_reward
                current_control_state = policy_step.control_state
                current_gas_level = policy_step.gas_level
                boost_lamp_level = policy_step.boost_lamp_level
                current_policy_action = policy_step.policy_action
                cnn_activations = policy_step.cnn_activations
                current_telemetry = policy_step.telemetry
                current_auxiliary_predictions = policy_step.auxiliary_predictions
                current_auxiliary_targets = policy_step.auxiliary_targets
                last_live_series_publish_time = policy_step.last_live_series_publish_time
                terminal_handled = controller.observe_step(
                    session=session,
                    info=info,
                )
                if terminal_handled and not controller.policy_owns_control():
                    terminal_info = controller.viewer_info(
                        info=info,
                        active_policy_control=active_policy_control,
                    )
                    record_controller_event(
                        controller=controller,
                        frame_recorder=frame_recorder,
                        info=terminal_info,
                    )
                    track_record_book = track_record_book.update(
                        info,
                        current_telemetry,
                        episode_done=True,
                    )
                    control_rate.reset()
                    episode += 1
                    raw_observation = None
                    observation = None
                    episode_reward = 0.0
                    live_series = EpisodeLiveSeriesTracker()
                    last_live_series_publish_time = 0.0
                    raw_info, info, current_telemetry = _fresh_menu_runtime_state(session)
                    info = controller.viewer_info(
                        info=info,
                        active_policy_control=None,
                    )
                    reset_info = dict(info)
                    active_policy_control = None
                    active_policy_started = False
                    current_policy_action = None
                    current_control_state = RaceControlState()
                    current_gas_level = 0.0
                    boost_lamp_level = 0.0
                    cnn_activations = None
                    current_auxiliary_predictions = None
                    current_auxiliary_targets = None
                    manual_control_enabled = False
                    publish_snapshot(policy_visible=False)
                    if not controller.has_active_attempt():
                        raise _CareerModeWorkerQuit()

            if current_step_seconds is not None:
                now = time.perf_counter()
                next_step_time = max(next_step_time + current_step_seconds, now)
    except _CareerModeWorkerQuit:
        raise
    except Exception as exc:
        raise RuntimeError(
            _career_runtime_error_context(
                exc,
                controller=controller,
                info=info,
                last_menu_step=last_menu_step,
            )
        ) from exc


def _career_runtime_error_context(
    exc: Exception,
    *,
    controller: CareerModeController,
    info: dict[str, object],
    last_menu_step: RawMenuStep | None,
) -> str:
    context = controller.debug_context(info)
    message = f"{exc}; {context}"
    if last_menu_step is None:
        return message
    return (
        f"{message}; last_step="
        f"{last_menu_step.phase}:{last_menu_step.menu_input}:{last_menu_step.frames}f"
    )


def _fresh_menu_runtime_state(
    session: CareerModeRuntimeSession,
) -> tuple[dict[str, object], dict[str, object], FZeroXTelemetry | None]:
    raw_info = menu_viewer_info(session)
    info = dict(raw_info)
    telemetry = _read_live_telemetry(session.emulator)
    return raw_info, info, telemetry


def _should_observe_policy_transition(
    *,
    policy_owns_control: bool,
    active_policy_started: bool,
    info: dict[str, object],
) -> bool:
    if not policy_owns_control:
        return False
    if active_policy_started:
        return True
    return not in_gp_race(info)
