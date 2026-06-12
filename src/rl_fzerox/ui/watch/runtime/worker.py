# src/rl_fzerox/ui/watch/runtime/worker.py
from __future__ import annotations

import os
import time
from multiprocessing.queues import Queue as ProcessQueue
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from fzerox_emulator import RaceControlState, SpinRequest
from rl_fzerox.core.envs.actions import ActionValue
from rl_fzerox.core.envs.engine.controls import action_mask_violations
from rl_fzerox.core.envs.telemetry import telemetry_boost_active
from rl_fzerox.core.runtime_spec.schema import WatchAppConfig
from rl_fzerox.ui.watch.live_series import (
    LIVE_SERIES_PUBLISH_POLICY,
    EpisodeLiveSeriesTracker,
)
from rl_fzerox.ui.watch.records import TrackRecordBook
from rl_fzerox.ui.watch.runtime.baseline import _save_baseline_state
from rl_fzerox.ui.watch.runtime.cnn import (
    DEFAULT_CNN_ACTIVATION_NORMALIZATION,
    CnnActivationSampler,
)
from rl_fzerox.ui.watch.runtime.course_commands import (
    apply_course_navigation_commands,
    next_watch_reset_after_episode,
)
from rl_fzerox.ui.watch.runtime.course_navigation import (
    WatchCourseRotation,
    sync_watch_rotation_info,
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
    _persist_reload_error,
    _policy_reload_error,
    _reset_policy_runner,
    _sync_policy_curriculum_stage,
)
from rl_fzerox.ui.watch.runtime.session import (
    load_watch_engine_tuning_state,
    open_watch_runtime_session,
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
from rl_fzerox.ui.watch.runtime.track_sampling import (
    ManagedTrackSamplingRefresh,
    missing_generated_x_cup_baseline_paths,
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

if TYPE_CHECKING:
    from rl_fzerox.core.policy.auxiliary_state import AuxiliaryStateTargetName

__all__ = (
    "_refresh_paused_cnn_activations",
    "run_simulation_worker",
)


class _SequentialResetEnv(Protocol):
    def set_next_sequential_reset_course(self, course_id: str | None) -> None: ...


def run_simulation_worker(
    config: WatchAppConfig,
    command_queue: ProcessQueue,
    snapshot_queue: ProcessQueue,
) -> None:
    """Run emulator and policy stepping in a process owned by the watch UI."""

    try:
        _run_simulation_loop(config, command_queue, snapshot_queue)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        publish_worker_message(snapshot_queue, WorkerError(message=str(exc)))
    finally:
        publish_worker_message(snapshot_queue, WorkerClosed())


def _run_simulation_loop(
    config: WatchAppConfig,
    command_queue: ProcessQueue,
    snapshot_queue: ProcessQueue,
) -> None:
    try:
        session = open_watch_runtime_session(config)
    except Exception as exc:
        raise RuntimeError(_watch_bootstrap_error_message(config, exc)) from exc
    emulator = session.emulator
    env = session.env
    policy_runner = session.policy_runner
    try:
        native_control_fps = session.native_control_fps
        target_control_fps = session.target_control_fps
        target_control_seconds = session.target_control_seconds
        control_rate = RateMeter(window=60)
        last_logged_reload_error: str | None = None
        episode = 0
        track_record_book = TrackRecordBook()
        paused = False
        deterministic_policy = bool(config.watch.deterministic_policy)
        manual_control_enabled = policy_runner is None
        manual_control_state = RaceControlState()
        manual_spin_request: SpinRequest = "none"
        committed_policy_action: ActionValue | None = None
        committed_action_mask_branches = env.action_mask_branches()
        cnn_visualization_enabled = False
        auxiliary_visualization_enabled = False
        live_visualization_enabled = False
        cnn_normalization = DEFAULT_CNN_ACTIVATION_NORMALIZATION
        cnn_sampler = CnnActivationSampler(refresh_interval_steps=1)
        active_track_sampling = config.env.track_sampling
        course_rotation = WatchCourseRotation.from_entries(active_track_sampling.entries)
        selected_reset_target_key = course_rotation.normalized_key(None)
        persistent_locked_reset_target_key: str | None = None
        track_sampling_refresh = ManagedTrackSamplingRefresh.from_config(config)
        watch_zeroed_state_features = session.watch_zeroed_state_features
        live_series = EpisodeLiveSeriesTracker()
        last_live_series_publish_time = 0.0
        auxiliary_target_names: tuple[AuxiliaryStateTargetName, ...] = (
            session.auxiliary_target_names
        )

        def publish_snapshot() -> None:
            nonlocal last_live_series_publish_time

            live_episode_series = None
            if live_visualization_enabled:
                current_time = time.perf_counter()
                if (
                    current_time - last_live_series_publish_time
                    >= LIVE_SERIES_PUBLISH_POLICY.interval_seconds
                ):
                    live_episode_series = live_series.snapshot()
                    last_live_series_publish_time = current_time
            publish_worker_message(
                snapshot_queue,
                _build_snapshot(
                    config=config,
                    env=env,
                    emulator=emulator,
                    observation=observation,
                    info=info,
                    reset_info=reset_info,
                    episode=episode,
                    episode_reward=episode_reward,
                    control_fps=control_rate.rate_hz(),
                    target_control_fps=target_control_fps,
                    control_state=current_control_state,
                    gas_level=current_gas_level,
                    boost_lamp_level=boost_lamp_level,
                    action_mask_branches=committed_action_mask_branches,
                    policy_action=current_policy_action,
                    policy_runner=policy_runner,
                    policy_auxiliary_state_predictions=current_auxiliary_predictions,
                    policy_auxiliary_state_targets=current_auxiliary_targets,
                    include_auxiliary_state=auxiliary_visualization_enabled,
                    auxiliary_target_names=auxiliary_target_names,
                    deterministic_policy=deterministic_policy,
                    manual_control_enabled=manual_control_enabled,
                    policy_reload_error=policy_reload_error,
                    cnn_activations=cnn_activations,
                    active_track_sampling=active_track_sampling,
                    track_record_book=track_record_book,
                    live_episode_series=live_episode_series,
                ),
            )

        def refresh_track_sampling(*, force: bool = False) -> bool:
            nonlocal active_track_sampling
            nonlocal course_rotation
            nonlocal persistent_locked_reset_target_key
            nonlocal selected_reset_target_key

            if track_sampling_refresh is None:
                return False
            refreshed_track_sampling = track_sampling_refresh.refreshed_config(
                active_track_sampling,
                force=force,
            )
            if refreshed_track_sampling is None:
                return False

            env.set_track_sampling_config(refreshed_track_sampling)
            active_track_sampling = refreshed_track_sampling
            course_rotation = WatchCourseRotation.from_entries(active_track_sampling.entries)
            selected_reset_target_key = course_rotation.normalized_key(selected_reset_target_key)
            if course_rotation.target_by_key(persistent_locked_reset_target_key) is None:
                persistent_locked_reset_target_key = None
                env.set_locked_reset_course(None)
            return True

        def track_sampling_ready_for_reset(*, force: bool = False) -> bool:
            nonlocal active_track_sampling
            nonlocal course_rotation
            nonlocal persistent_locked_reset_target_key
            nonlocal selected_reset_target_key

            if track_sampling_refresh is None:
                return True
            status = track_sampling_refresh.refresh_status(active_track_sampling, force=force)
            if status.refreshed_config is not None:
                env.set_track_sampling_config(status.refreshed_config)
                active_track_sampling = status.refreshed_config
                course_rotation = WatchCourseRotation.from_entries(active_track_sampling.entries)
                selected_reset_target_key = course_rotation.normalized_key(
                    selected_reset_target_key
                )
                if course_rotation.target_by_key(persistent_locked_reset_target_key) is None:
                    persistent_locked_reset_target_key = None
                    env.set_locked_reset_course(None)
            return status.ready_for_reset and not missing_generated_x_cup_baseline_paths(
                active_track_sampling,
            )

        while config.watch.episodes is None or episode < config.watch.episodes:
            force_track_sampling_check = True
            while not track_sampling_ready_for_reset(force=force_track_sampling_check):
                force_track_sampling_check = False
                time.sleep(0.25)
            load_watch_engine_tuning_state(config, env)
            env.set_engine_tuning_selection("greedy" if deterministic_policy else "sample")
            env.set_locked_reset_course(persistent_locked_reset_target_key)
            if persistent_locked_reset_target_key is None and selected_reset_target_key is not None:
                env.set_next_sequential_reset_course(selected_reset_target_key)
            reset_seed = config.seed if episode == 0 else None
            raw_observation, raw_info = env.reset(seed=reset_seed)
            observation, info = apply_watch_state_feature_zeroing(
                raw_observation,
                raw_info,
                watch_zeroed_features=watch_zeroed_state_features,
            )
            current_target = course_rotation.target_for_info(info)
            if current_target is not None:
                selected_reset_target_key = current_target.key
            _reset_policy_runner(policy_runner)
            reset_info = dict(info)
            sync_watch_rotation_info(
                info=info,
                reset_info=reset_info,
                rotation=course_rotation,
                selected_reset_target_key=selected_reset_target_key,
                locked_reset_target_key=persistent_locked_reset_target_key,
            )
            current_control_state = env.last_requested_control_state
            current_gas_level = env.last_gas_level
            committed_action_mask_branches = env.action_mask_branches()
            current_telemetry = _read_live_telemetry(emulator)
            current_auxiliary_predictions = _current_auxiliary_predictions(
                policy_runner=policy_runner,
                enabled=auxiliary_visualization_enabled,
                observation=observation,
                target_names=auxiliary_target_names,
            )
            current_auxiliary_targets = _current_auxiliary_targets(
                telemetry=current_telemetry,
                enabled=auxiliary_visualization_enabled,
                target_names=auxiliary_target_names,
            )
            boost_lamp_level = 0.0
            current_policy_action: ActionValue | None = committed_policy_action
            cnn_activations = None
            terminated = False
            truncated = False
            episode_reward = 0.0
            next_step_time = time.perf_counter()
            policy_reload_error = _policy_reload_error(policy_runner)
            live_series.observe_decision(
                episode=episode,
                info=info,
                episode_reward=episode_reward,
                telemetry_data=_telemetry_to_data(current_telemetry),
                action_repeat=config.env.action_repeat,
            )

            publish_snapshot()

            while not (terminated or truncated):
                previous_cnn_visualization_enabled = cnn_visualization_enabled
                previous_auxiliary_visualization_enabled = auxiliary_visualization_enabled
                previous_live_visualization_enabled = live_visualization_enabled
                previous_cnn_normalization = cnn_normalization
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
                manual_control_enabled = (
                    True if policy_runner is None else commands.manual_control_enabled
                )
                if manual_control_enabled:
                    manual_spin_request = commands.spin_request
                else:
                    manual_spin_request = "none"
                cnn_visualization_enabled = commands.cnn_visualization_enabled
                auxiliary_visualization_enabled = (
                    bool(auxiliary_target_names) and commands.auxiliary_visualization_enabled
                )
                live_visualization_enabled = commands.live_visualization_enabled
                cnn_normalization = commands.cnn_normalization
                if commands.quit_requested:
                    return
                if live_visualization_enabled != previous_live_visualization_enabled:
                    last_live_series_publish_time = 0.0
                    publish_snapshot()
                if refresh_track_sampling() and commands.paused:
                    publish_snapshot()
                if commands.toggle_zeroed_state_feature_name is not None:
                    watch_zeroed_state_features = toggle_watch_state_feature(
                        watch_zeroed_state_features,
                        commands.toggle_zeroed_state_feature_name,
                    )
                    observation, info = apply_watch_state_feature_zeroing(
                        raw_observation,
                        raw_info,
                        watch_zeroed_features=watch_zeroed_state_features,
                    )
                    publish_snapshot()
                course_command = apply_course_navigation_commands(
                    commands,
                    env=env,
                    info=info,
                    reset_info=reset_info,
                    rotation=course_rotation,
                    selected_reset_target_key=selected_reset_target_key,
                    locked_reset_target_key=persistent_locked_reset_target_key,
                )
                selected_reset_target_key = course_command.selected_reset_target_key
                persistent_locked_reset_target_key = course_command.locked_reset_target_key
                if course_command.reset_requested:
                    break
                if course_command.lock_state_changed:
                    publish_snapshot()
                if commands.reset_control_fps:
                    target_control_fps = native_control_fps
                    target_control_seconds = _target_seconds(target_control_fps)
                    control_rate.trim_to_recent()
                    next_step_time = time.perf_counter()
                elif commands.control_fps_delta:
                    target_control_fps = _adjust_control_fps(
                        target_control_fps,
                        commands.control_fps_delta,
                        native_control_fps=native_control_fps,
                    )
                    target_control_seconds = _target_seconds(target_control_fps)
                    control_rate.trim_to_recent()
                    next_step_time = time.perf_counter()
                if commands.save_requests:
                    _save_baseline_state(
                        emulator=emulator,
                        baseline_state_path=config.emulator.baseline_state_path,
                    )
                if commands.toggle_deterministic_policy and policy_runner is not None:
                    deterministic_policy = not deterministic_policy
                    env.set_engine_tuning_selection(
                        "greedy" if deterministic_policy else "sample"
                    )
                if manual_control_enabled:
                    current_control_state = commands.control_state

                policy_reload_error = _policy_reload_error(policy_runner)
                last_logged_reload_error = _persist_reload_error(
                    reload_error=policy_reload_error,
                    runtime_dir=config.emulator.runtime_dir,
                    last_logged_reload_error=last_logged_reload_error,
                )
                if (
                    auxiliary_visualization_enabled != previous_auxiliary_visualization_enabled
                    or commands.toggle_zeroed_state_feature_name is not None
                ):
                    current_auxiliary_predictions = _current_auxiliary_predictions(
                        policy_runner=policy_runner,
                        enabled=auxiliary_visualization_enabled,
                        observation=observation,
                        target_names=auxiliary_target_names,
                    )
                    current_auxiliary_targets = _current_auxiliary_targets(
                        telemetry=current_telemetry,
                        enabled=auxiliary_visualization_enabled,
                        target_names=auxiliary_target_names,
                    )

                if commands.paused and commands.step_requests <= 0:
                    cnn_activations, cnn_snapshot_changed = _refresh_paused_cnn_activations(
                        current_activations=cnn_activations,
                        cnn_sampler=cnn_sampler,
                        cnn_visualization_enabled=cnn_visualization_enabled,
                        previous_cnn_visualization_enabled=previous_cnn_visualization_enabled,
                        cnn_normalization=cnn_normalization,
                        previous_cnn_normalization=previous_cnn_normalization,
                        policy_runner=policy_runner,
                        observation=observation,
                    )
                    if (
                        cnn_snapshot_changed
                        or auxiliary_visualization_enabled
                        != previous_auxiliary_visualization_enabled
                    ):
                        publish_snapshot()
                    time.sleep(0.01)
                    continue
                if not commands.paused and target_control_seconds is not None:
                    wait_seconds = next_step_time - time.perf_counter()
                    if wait_seconds > 0.0:
                        time.sleep(min(wait_seconds, 0.005))
                        continue

                single_frame_manual = commands.paused and commands.step_requests > 0
                previous_observation = observation
                previous_info = dict(info)
                previous_episode_reward = episode_reward
                previous_telemetry = _read_live_telemetry(emulator)
                previous_control_state = current_control_state
                previous_gas_level = current_gas_level
                previous_policy_action = committed_policy_action
                previous_action_mask_branches = committed_action_mask_branches
                previous_auxiliary_predictions = current_auxiliary_predictions
                previous_auxiliary_targets = current_auxiliary_targets
                decision_action_mask = env.action_mask_snapshot()
                if manual_control_enabled:
                    if single_frame_manual:
                        observation, reward, terminated, truncated, info = env.step_frame(
                            current_control_state,
                            spin_request=manual_spin_request,
                        )
                        display_frames = (env.render(),)
                        display_controller_masks = (current_control_state.control_mask,)
                    else:
                        watch_step = env.step_control_watch(
                            current_control_state,
                            spin_request=manual_spin_request,
                        )
                        observation, reward, terminated, truncated, info = watch_step.gym_result()
                        display_frames = watch_step.display_frames
                        display_controller_masks = watch_step.display_controller_masks
                    current_policy_action = None
                    current_control_state = env.last_requested_control_state
                    current_gas_level = env.last_gas_level
                    committed_policy_action = None
                    cnn_activations = None
                else:
                    assert policy_runner is not None
                    _sync_policy_curriculum_stage(policy_runner, env)
                    decision_action_mask = env.action_mask_snapshot()
                    policy_action_mask = (
                        decision_action_mask if policy_runner.supports_action_masks else None
                    )
                    action = policy_runner.predict(
                        observation,
                        deterministic=deterministic_policy,
                        action_masks=(
                            policy_action_mask.flat if policy_action_mask is not None else None
                        ),
                        refresh=False,
                    )
                    if policy_action_mask is not None:
                        violations = action_mask_violations(policy_action_mask.branches, action)
                        if violations:
                            details = ", ".join(violations)
                            raise RuntimeError(f"Policy selected masked action values: {details}")
                    current_policy_action = action
                    cnn_activations = cnn_sampler.capture(
                        enabled=commands.cnn_visualization_enabled,
                        policy_runner=policy_runner,
                        observation=observation,
                        normalization=commands.cnn_normalization,
                    )
                    watch_step = env.step_watch(action)
                    raw_observation, reward, terminated, truncated, raw_info = (
                        watch_step.gym_result()
                    )
                    display_frames = watch_step.display_frames
                    display_controller_masks = watch_step.display_controller_masks
                    current_control_state = env.last_requested_control_state
                    current_gas_level = env.last_gas_level
                if manual_control_enabled:
                    raw_observation = observation
                    raw_info = info
                observation, info = apply_watch_state_feature_zeroing(
                    raw_observation,
                    raw_info,
                    watch_zeroed_features=watch_zeroed_state_features,
                )
                final_action_mask_branches = env.action_mask_branches()
                live_telemetry = _read_live_telemetry(emulator)
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
                boost_lamp_level = _next_boost_lamp_level(
                    previous=boost_lamp_level,
                    control_state=current_control_state,
                    boost_active=telemetry_boost_active(live_telemetry),
                    action_repeat=config.env.action_repeat,
                )
                control_rate.tick()
                episode_reward += reward
                live_series.observe_decision(
                    episode=episode,
                    info=info,
                    episode_reward=episode_reward,
                    telemetry_data=_telemetry_to_data(live_telemetry),
                    action_repeat=config.env.action_repeat,
                )
                track_record_book = track_record_book.update(
                    info,
                    live_telemetry,
                    episode_done=terminated or truncated,
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
                    config=config,
                    env=env,
                    emulator=emulator,
                    snapshot_queue=snapshot_queue,
                    display_frames=display_frames,
                    display_controller_masks=display_controller_masks,
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
                    previous_policy_action=previous_policy_action,
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
                    target_control_fps=target_control_fps,
                    target_control_seconds=target_control_seconds,
                    boost_lamp_level=boost_lamp_level,
                    policy_runner=policy_runner,
                    deterministic_policy=deterministic_policy,
                    policy_reload_error=policy_reload_error,
                    cnn_activations=cnn_activations,
                    active_track_sampling=active_track_sampling,
                    track_record_book=track_record_book,
                    manual_control_enabled=manual_control_enabled,
                    live_episode_series=live_episode_series,
                )
                committed_policy_action = current_policy_action
                committed_action_mask_branches = final_action_mask_branches
                current_telemetry = live_telemetry
                current_auxiliary_predictions = final_auxiliary_predictions
                current_auxiliary_targets = final_auxiliary_targets
                if target_control_seconds is not None:
                    now = time.perf_counter()
                    next_step_time = max(next_step_time + target_control_seconds, now)
            selected_reset_target_key = _sync_next_watch_reset_after_episode(
                env=env,
                rotation=course_rotation,
                info=info,
                episode_done=terminated or truncated,
                selected_reset_target_key=selected_reset_target_key,
                locked_reset_target_key=persistent_locked_reset_target_key,
            )
            episode += 1
    finally:
        session.close()


def _sync_next_watch_reset_after_episode(
    *,
    env: _SequentialResetEnv,
    rotation: WatchCourseRotation,
    info: dict[str, object],
    episode_done: bool,
    selected_reset_target_key: str | None,
    locked_reset_target_key: str | None,
) -> str | None:
    if not episode_done:
        return selected_reset_target_key
    next_target_key = next_watch_reset_after_episode(
        rotation=rotation,
        info=info,
        episode_done=episode_done,
        selected_reset_target_key=selected_reset_target_key,
        locked_reset_target_key=locked_reset_target_key,
    )
    if locked_reset_target_key is None and next_target_key is not None:
        env.set_next_sequential_reset_course(next_target_key)
    return next_target_key


def _watch_bootstrap_error_message(config: WatchAppConfig, error: Exception) -> str:
    track_sampling = config.env.track_sampling
    return "\n".join(
        (
            f"watch worker failed during bootstrap: {type(error).__name__}: {error}",
            f"core_path={_path_report(config.emulator.core_path)}",
            f"rom_path={_path_report(config.emulator.rom_path)}",
            f"runtime_dir={_path_report(config.emulator.runtime_dir)}",
            f"baseline_state_path={_path_report(config.emulator.baseline_state_path)}",
            f"renderer={config.emulator.renderer}",
            "track_sampling="
            f"enabled={track_sampling.enabled} entries={len(track_sampling.entries)}",
        )
    )


def _path_report(path: Path | None) -> str:
    if path is None:
        return "-"
    resolved = path.expanduser().resolve()
    if resolved.is_file():
        return (
            f"{resolved} file size={resolved.stat().st_size} "
            f"readable={_yes_no(os.access(resolved, os.R_OK))}"
        )
    if resolved.is_dir():
        return (
            f"{resolved} dir readable={_yes_no(os.access(resolved, os.R_OK))} "
            f"writable={_yes_no(os.access(resolved, os.W_OK))}"
        )
    return f"{resolved} missing parent_exists={_yes_no(resolved.parent.exists())}"


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"
