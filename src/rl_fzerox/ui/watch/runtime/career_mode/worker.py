# src/rl_fzerox/ui/watch/runtime/career_mode/worker.py
from __future__ import annotations

import time
from collections.abc import Mapping
from multiprocessing.queues import Queue as ProcessQueue
from typing import TypeAlias

from fzerox_emulator import FZeroXTelemetry, RaceControlState, SpinRequest
from rl_fzerox.core.career_mode.runner.controller import CareerModeController
from rl_fzerox.core.career_mode.runner.menu import (
    MenuInput,
    RawMenuStep,
    course_id_from_info,
    in_gp_race,
)
from rl_fzerox.core.career_mode.runner.policy import CareerModePolicyControl
from rl_fzerox.core.career_mode.runner.save_file import (
    load_save_ram,
    persist_save_ram,
    save_game_id_from_config,
    store_from_config,
)
from rl_fzerox.core.envs.actions import ActionValue
from rl_fzerox.core.envs.engine.controls import action_mask_violations
from rl_fzerox.core.envs.observations import ObservationValue
from rl_fzerox.core.envs.telemetry import telemetry_boost_active
from rl_fzerox.core.policy.auxiliary_state import AuxiliaryStateTargetName
from rl_fzerox.core.runtime_spec.schema import WatchAppConfig
from rl_fzerox.core.training.inference import PolicyRunner
from rl_fzerox.ui.watch.live_series import (
    LIVE_SERIES_PUBLISH_POLICY,
    EpisodeLiveSeriesTracker,
)
from rl_fzerox.ui.watch.runtime.career_mode.attempts import (
    RUNNER_CLOSED_REASON,
    RUNNER_FAILED_REASON,
    fail_running_attempts,
)
from rl_fzerox.ui.watch.runtime.career_mode.menu import (
    menu_viewer_info,
    reset_race_progress_info,
    step_menu,
)
from rl_fzerox.ui.watch.runtime.career_mode.session import (
    CareerModeRuntimeSession,
    open_career_mode_runtime_session,
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
from rl_fzerox.ui.watch.runtime.cnn import (
    DEFAULT_CNN_ACTIVATION_NORMALIZATION,
    CnnActivationNormalizationMode,
    CnnActivationSampler,
    CnnActivationSnapshot,
)
from rl_fzerox.ui.watch.runtime.ipc import (
    WorkerClosed,
    WorkerError,
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
from rl_fzerox.ui.watch.runtime.snapshots import (
    _build_snapshot,
    _next_boost_lamp_level,
    _publish_step_snapshots,
)
from rl_fzerox.ui.watch.runtime.telemetry import _read_live_telemetry, _telemetry_to_data
from rl_fzerox.ui.watch.runtime.timing import (
    RateMeter,
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
    load_save_ram(config, session)

    failure_reason = RUNNER_CLOSED_REASON
    try:
        _run_loaded_career_mode_loop(
            config=config,
            session=session,
            controller=controller,
            command_queue=command_queue,
            snapshot_queue=snapshot_queue,
        )
    except BaseException:
        failure_reason = RUNNER_FAILED_REASON
        raise
    finally:
        _close_career_mode(config, session, failure_reason=failure_reason)
        session.close()


def _run_loaded_career_mode_loop(
    *,
    config: WatchAppConfig,
    session: CareerModeRuntimeSession,
    controller: CareerModeController,
    command_queue: ProcessQueue,
    snapshot_queue: ProcessQueue,
) -> None:
    native_control_fps = session.native_control_fps
    target_control_fps = session.target_control_fps
    target_control_seconds = session.target_control_seconds
    native_frame_seconds = _native_frame_seconds(target_control_seconds)
    control_rate = RateMeter(window=60)
    watch_zeroed_state_features = session.watch_zeroed_state_features
    auxiliary_target_names = session.auxiliary_target_names

    paused = False
    deterministic_policy = bool(config.watch.deterministic_policy)
    manual_control_enabled = False
    manual_control_state = RaceControlState()
    current_control_state = RaceControlState()
    current_gas_level = 0.0
    boost_lamp_level = 0.0
    episode = 0
    episode_reward = 0.0
    cnn_visualization_enabled = False
    auxiliary_visualization_enabled = False
    live_visualization_enabled = False
    live_series = EpisodeLiveSeriesTracker()
    last_live_series_publish_time = 0.0
    cnn_normalization = DEFAULT_CNN_ACTIVATION_NORMALIZATION
    cnn_sampler = CnnActivationSampler(refresh_interval_steps=1)
    cnn_activations = None
    active_policy_control: CareerModePolicyControl | None = None
    active_policy_started = False
    current_policy_action: ActionValue | None = None

    raw_observation: ObservationValue | None = None
    observation: ObservationValue | None = None
    raw_info = menu_viewer_info(session)
    info = controller.viewer_info(
        info=dict(raw_info),
        active_policy_control=None,
    )
    reset_info = dict(info)
    last_menu_step: RawMenuStep | None = None
    current_telemetry = _read_live_telemetry(session.emulator)
    current_auxiliary_predictions = None
    current_auxiliary_targets = None

    publish_worker_message(
        snapshot_queue,
        _build_snapshot(
            config=config,
            env=session,
            emulator=session.emulator,
            observation=None,
            info=info,
            reset_info=reset_info,
            episode=episode,
            episode_reward=episode_reward,
            control_fps=control_rate.rate_hz(),
            target_control_fps=target_control_fps,
            action_repeat=1,
            control_state=current_control_state,
            gas_level=current_gas_level,
            boost_lamp_level=boost_lamp_level,
            action_mask_branches=session.action_mask_branches(),
            policy_action=None,
            policy_runner=None,
            policy_auxiliary_state_predictions=None,
            policy_auxiliary_state_targets=None,
            include_auxiliary_state=False,
            auxiliary_target_names=auxiliary_target_names,
            deterministic_policy=deterministic_policy,
            manual_control_enabled=False,
            policy_reload_error=None,
            cnn_activations=None,
            best_finish_position=None,
            best_finish_ranks={},
            best_finish_times={},
            latest_finish_times={},
            latest_finish_deltas_ms={},
            failed_track_attempts=frozenset(),
        ),
    )
    next_step_time = time.perf_counter()

    try:
        _run_career_mode_loop_body(
            config=config,
            session=session,
            controller=controller,
            command_queue=command_queue,
            snapshot_queue=snapshot_queue,
            control_rate=control_rate,
            native_control_fps=native_control_fps,
            target_control_fps=target_control_fps,
            target_control_seconds=target_control_seconds,
            native_frame_seconds=native_frame_seconds,
            next_step_time=next_step_time,
            paused=paused,
            deterministic_policy=deterministic_policy,
            manual_control_enabled=manual_control_enabled,
            manual_control_state=manual_control_state,
            current_control_state=current_control_state,
            current_gas_level=current_gas_level,
            boost_lamp_level=boost_lamp_level,
            episode=episode,
            episode_reward=episode_reward,
            cnn_visualization_enabled=cnn_visualization_enabled,
            auxiliary_visualization_enabled=auxiliary_visualization_enabled,
            live_visualization_enabled=live_visualization_enabled,
            live_series=live_series,
            last_live_series_publish_time=last_live_series_publish_time,
            cnn_normalization=cnn_normalization,
            cnn_sampler=cnn_sampler,
            watch_zeroed_state_features=watch_zeroed_state_features,
            auxiliary_target_names=auxiliary_target_names,
            active_policy_control=active_policy_control,
            active_policy_started=active_policy_started,
            current_policy_action=current_policy_action,
            raw_observation=raw_observation,
            observation=observation,
            raw_info=raw_info,
            info=info,
            reset_info=reset_info,
            current_telemetry=current_telemetry,
            current_auxiliary_predictions=current_auxiliary_predictions,
            current_auxiliary_targets=current_auxiliary_targets,
            cnn_activations=cnn_activations,
            last_menu_step=last_menu_step,
        )
    except _CareerModeWorkerQuit:
        return


class _CareerModeWorkerQuit(Exception):
    """Internal signal used to unwind the Career Mode worker loop."""


def _run_career_mode_loop_body(
    *,
    config: WatchAppConfig,
    session: CareerModeRuntimeSession,
    controller: CareerModeController,
    command_queue: ProcessQueue,
    snapshot_queue: ProcessQueue,
    control_rate: RateMeter,
    native_control_fps: float,
    target_control_fps: float | None,
    target_control_seconds: float | None,
    native_frame_seconds: float | None,
    next_step_time: float,
    paused: bool,
    deterministic_policy: bool,
    manual_control_enabled: bool,
    manual_control_state: RaceControlState,
    current_control_state: RaceControlState,
    current_gas_level: float,
    boost_lamp_level: float,
    episode: int,
    episode_reward: float,
    cnn_visualization_enabled: bool,
    auxiliary_visualization_enabled: bool,
    live_visualization_enabled: bool,
    live_series: EpisodeLiveSeriesTracker,
    last_live_series_publish_time: float,
    cnn_normalization: CnnActivationNormalizationMode,
    cnn_sampler: CnnActivationSampler,
    watch_zeroed_state_features: frozenset[str],
    auxiliary_target_names: tuple[AuxiliaryStateTargetName, ...],
    active_policy_control: CareerModePolicyControl | None,
    active_policy_started: bool,
    current_policy_action: ActionValue | None,
    raw_observation: ObservationValue | None,
    observation: ObservationValue | None,
    raw_info: dict[str, object],
    info: dict[str, object],
    reset_info: dict[str, object],
    current_telemetry: FZeroXTelemetry | None,
    current_auxiliary_predictions: dict[str, object] | None,
    current_auxiliary_targets: dict[str, object] | None,
    cnn_activations: CnnActivationSnapshot | None,
    last_menu_step: RawMenuStep | None,
) -> None:
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
                best_finish_position=None,
                best_finish_ranks={},
                best_finish_times={},
                latest_finish_times={},
                latest_finish_deltas_ms={},
                failed_track_attempts=frozenset(),
            ),
        )

    try:
        while True:
            previous_cnn_visualization_enabled = cnn_visualization_enabled
            previous_auxiliary_visualization_enabled = auxiliary_visualization_enabled
            previous_live_visualization_enabled = live_visualization_enabled
            previous_cnn_normalization = cnn_normalization
            commands, paused, manual_control_state = drain_worker_commands(
                command_queue,
                paused=paused,
                control_state=manual_control_state,
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

            if controller.before_step(session=session, info=info):
                raw_info, info, current_telemetry = _fresh_menu_runtime_state(session)
                info = controller.viewer_info(
                    info=info,
                    active_policy_control=active_policy_control,
                )
                reset_info = dict(info)
            if active_policy_started and controller.policy_owns_control():
                terminal_handled = controller.observe_step(
                    session=session,
                    info=info,
                )
                if not terminal_handled and not in_gp_race(info):
                    raise RuntimeError("Career Mode left a race before observing a game result")
                if terminal_handled:
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
                    episode_reward = _required_episode_return(info)
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
                manual_control_enabled = commands.manual_control_enabled
                if manual_control_enabled:
                    current_control_state = commands.control_state
                (
                    raw_observation,
                    raw_info,
                    observation,
                    info,
                    episode_reward,
                    current_control_state,
                    current_gas_level,
                    boost_lamp_level,
                    current_policy_action,
                    cnn_activations,
                    current_telemetry,
                    current_auxiliary_predictions,
                    current_auxiliary_targets,
                    last_live_series_publish_time,
                ) = _step_policy_or_manual(
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
                    manual_control_enabled=manual_control_enabled,
                    current_control_state=current_control_state,
                    spin_request=commands.spin_request,
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
                )
                terminal_handled = controller.observe_step(
                    session=session,
                    info=info,
                )
                if terminal_handled and not controller.policy_owns_control():
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


_PolicyStepResult: TypeAlias = tuple[
    ObservationValue,
    dict[str, object],
    ObservationValue,
    dict[str, object],
    float,
    RaceControlState,
    float,
    float,
    ActionValue | None,
    CnnActivationSnapshot | None,
    FZeroXTelemetry | None,
    dict[str, object] | None,
    dict[str, object] | None,
    float,
]


def _step_policy_or_manual(
    *,
    config: WatchAppConfig,
    session: CareerModeRuntimeSession,
    controller: CareerModeController,
    snapshot_queue: ProcessQueue,
    active_policy_control: CareerModePolicyControl,
    policy_runner: PolicyRunner,
    observation: ObservationValue,
    info: dict[str, object],
    reset_info: dict[str, object],
    episode: int,
    episode_reward: float,
    control_rate: RateMeter,
    target_policy_control_fps: float | None,
    target_control_seconds: float | None,
    deterministic_policy: bool,
    manual_control_enabled: bool,
    current_control_state: RaceControlState,
    spin_request: SpinRequest,
    boost_lamp_level: float,
    cnn_visualization_enabled: bool,
    cnn_normalization: CnnActivationNormalizationMode,
    cnn_sampler: CnnActivationSampler,
    auxiliary_visualization_enabled: bool,
    auxiliary_target_names: tuple[AuxiliaryStateTargetName, ...],
    watch_zeroed_state_features: frozenset[str],
    live_visualization_enabled: bool,
    live_series: EpisodeLiveSeriesTracker,
    last_live_series_publish_time: float,
) -> _PolicyStepResult:
    previous_observation = observation
    previous_info = controller.viewer_info(
        info=dict(info),
        active_policy_control=active_policy_control,
    )
    previous_episode_reward = episode_reward
    previous_telemetry = _read_live_telemetry(session.emulator)
    previous_control_state = current_control_state
    previous_gas_level = session.last_gas_level
    previous_action_mask_branches = session.action_mask_branches()
    previous_auxiliary_predictions = _current_auxiliary_predictions(
        policy_runner=policy_runner,
        enabled=auxiliary_visualization_enabled,
        observation=observation,
        target_names=auxiliary_target_names,
    )
    previous_auxiliary_targets = _current_auxiliary_targets(
        telemetry=previous_telemetry,
        enabled=auxiliary_visualization_enabled,
        target_names=auxiliary_target_names,
    )
    if manual_control_enabled:
        race_step = session.step_manual_race(
            current_control_state,
            spin_request=spin_request,
        )
        raw_observation = race_step.observation
        raw_info = race_step.info
        current_policy_action = None
        cnn_activations = None
    else:
        policy_runner.refresh_if_due(interval_seconds=10.0)
        session.sync_policy_curriculum_stage(policy_runner.checkpoint_curriculum_stage_index)
        decision_action_mask = session.action_mask_snapshot()
        policy_action_mask = decision_action_mask if policy_runner.supports_action_masks else None
        action = policy_runner.predict(
            observation,
            deterministic=deterministic_policy,
            action_masks=policy_action_mask.flat if policy_action_mask is not None else None,
            refresh=False,
        )
        if policy_action_mask is not None:
            violations = action_mask_violations(policy_action_mask.branches, action)
            if violations:
                details = ", ".join(violations)
                raise RuntimeError(f"Policy selected masked action values: {details}")
        current_policy_action = action
        cnn_activations = cnn_sampler.capture(
            enabled=cnn_visualization_enabled,
            policy_runner=policy_runner,
            observation=observation,
            normalization=cnn_normalization,
        )
        race_step = session.step_policy(action)
        raw_observation = race_step.observation
        raw_info = race_step.info

    observation, info = apply_watch_state_feature_zeroing(
        raw_observation,
        raw_info,
        watch_zeroed_features=watch_zeroed_state_features,
    )
    info = controller.viewer_info(
        info=info,
        active_policy_control=active_policy_control,
    )
    current_control_state = session.last_requested_control_state
    current_gas_level = session.last_gas_level
    final_action_mask_branches = session.action_mask_branches()
    live_telemetry = _read_live_telemetry(session.emulator)
    final_auxiliary_predictions = _current_auxiliary_predictions(
        policy_runner=policy_runner,
        enabled=auxiliary_visualization_enabled,
        observation=observation,
        target_names=auxiliary_target_names,
    )
    final_auxiliary_targets = _current_auxiliary_targets(
        telemetry=live_telemetry,
        enabled=auxiliary_visualization_enabled,
        target_names=auxiliary_target_names,
    )
    snapshot_config = session.snapshot_config(config)
    boost_lamp_level = _next_boost_lamp_level(
        previous=boost_lamp_level,
        control_state=current_control_state,
        boost_active=telemetry_boost_active(live_telemetry),
        action_repeat=snapshot_config.env.action_repeat,
    )
    control_rate.tick()
    episode_reward = _required_episode_return(info)
    measured_game_fps_value = measured_game_fps(
        control_fps=control_rate.rate_hz(),
        action_repeat=snapshot_config.env.action_repeat,
    )
    previous_info = with_measured_game_fps(
        previous_info,
        game_fps=measured_game_fps_value,
        game_fps_target=target_game_fps(
            target_control_fps=target_policy_control_fps,
            action_repeat=snapshot_config.env.action_repeat,
        ),
    )
    info = with_measured_game_fps(
        info,
        game_fps=measured_game_fps_value,
        game_fps_target=target_game_fps(
            target_control_fps=target_policy_control_fps,
            action_repeat=snapshot_config.env.action_repeat,
        ),
    )
    live_series.observe_decision(
        episode=episode,
        info=info,
        episode_reward=episode_reward,
        telemetry_data=_telemetry_to_data(live_telemetry),
        action_repeat=snapshot_config.env.action_repeat,
    )
    live_episode_series = None
    if live_visualization_enabled:
        current_time = time.perf_counter()
        if (
            current_time - last_live_series_publish_time
            >= LIVE_SERIES_PUBLISH_POLICY.interval_seconds
        ):
            live_episode_series = live_series.snapshot()
            last_live_series_publish_time = current_time
    _publish_step_snapshots(
        config=snapshot_config,
        env=session,
        emulator=session.emulator,
        snapshot_queue=snapshot_queue,
        display_frames=race_step.display_frames,
        display_controller_masks=race_step.display_controller_masks,
        previous_observation=previous_observation,
        previous_info=previous_info,
        previous_episode_reward=previous_episode_reward,
        previous_telemetry=previous_telemetry,
        final_observation=observation,
        final_info=info,
        final_episode_reward=episode_reward,
        final_telemetry=live_telemetry,
        previous_control_state=previous_control_state,
        previous_gas_level=previous_gas_level,
        previous_action_mask_branches=previous_action_mask_branches,
        previous_policy_action=None,
        final_control_state=current_control_state,
        final_gas_level=current_gas_level,
        final_action_mask_branches=final_action_mask_branches,
        final_policy_action=current_policy_action,
        previous_auxiliary_predictions=previous_auxiliary_predictions,
        previous_auxiliary_targets=previous_auxiliary_targets,
        final_auxiliary_predictions=final_auxiliary_predictions,
        final_auxiliary_targets=final_auxiliary_targets,
        reset_info=reset_info,
        episode=episode,
        control_fps=control_rate.rate_hz(),
        target_control_fps=target_policy_control_fps,
        target_control_seconds=target_control_seconds,
        boost_lamp_level=boost_lamp_level,
        policy_runner=policy_runner,
        deterministic_policy=deterministic_policy,
        policy_reload_error=_policy_reload_error(policy_runner),
        cnn_activations=cnn_activations,
        active_track_sampling=None,
        best_finish_position=None,
        best_finish_ranks={},
        best_finish_times={},
        latest_finish_times={},
        latest_finish_deltas_ms={},
        failed_track_attempts=frozenset(),
        manual_control_enabled=manual_control_enabled,
        live_episode_series=live_episode_series,
    )
    return (
        raw_observation,
        raw_info,
        observation,
        info,
        episode_reward,
        current_control_state,
        current_gas_level,
        boost_lamp_level,
        current_policy_action,
        cnn_activations,
        live_telemetry,
        final_auxiliary_predictions,
        final_auxiliary_targets,
        last_live_series_publish_time,
    )


def _required_episode_return(info: Mapping[str, object]) -> float:
    value = info.get("episode_return")
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    raise RuntimeError("Career Mode policy step did not publish episode_return")


def _close_career_mode(
    config: WatchAppConfig,
    session: CareerModeRuntimeSession,
    *,
    failure_reason: str,
) -> None:
    persist_save_ram(config, session)
    store = store_from_config(config)
    save_game_id = save_game_id_from_config(config)
    fail_running_attempts(store, save_game_id=save_game_id, failure_reason=failure_reason)


def _mark_runner_failed(config: WatchAppConfig) -> None:
    store = store_from_config(config)
    save_game_id = config.watch.managed_save_game_id
    if save_game_id is not None:
        fail_running_attempts(
            store,
            save_game_id=save_game_id,
            failure_reason=RUNNER_FAILED_REASON,
        )
