# src/rl_fzerox/ui/watch/runtime/worker.py
from __future__ import annotations

import time
from multiprocessing.queues import Queue as ProcessQueue
from typing import TYPE_CHECKING

from fzerox_emulator import ControllerState, Emulator
from rl_fzerox.core.envs import FZeroXEnv
from rl_fzerox.core.envs.actions import ActionValue
from rl_fzerox.core.envs.engine.controls import action_mask_violations
from rl_fzerox.core.envs.telemetry import telemetry_boost_active
from rl_fzerox.core.runtime_spec.schema import TrackSamplingEntryConfig, WatchAppConfig
from rl_fzerox.core.seed import seed_process
from rl_fzerox.ui.watch.runtime.baseline import _save_baseline_state
from rl_fzerox.ui.watch.runtime.cnn import (
    DEFAULT_CNN_ACTIVATION_NORMALIZATION,
    CnnActivationNormalizationMode,
    CnnActivationSampler,
    CnnActivationSnapshot,
)
from rl_fzerox.ui.watch.runtime.episode import (
    _update_best_finish_position,
    _update_best_finish_times,
    _update_failed_track_attempts,
    _update_latest_finish_deltas_ms,
    _update_latest_finish_times,
)
from rl_fzerox.ui.watch.runtime.ipc import (
    WorkerClosed,
    WorkerError,
    drain_worker_commands,
    publish_worker_message,
)
from rl_fzerox.ui.watch.runtime.observation import (
    apply_watch_state_feature_zeroing,
    configured_watch_zeroed_features,
    toggle_watch_state_feature,
)
from rl_fzerox.ui.watch.runtime.policy import (
    _load_policy_runner,
    _persist_reload_error,
    _policy_reload_error,
    _reset_policy_runner,
    _sync_policy_curriculum_stage,
)
from rl_fzerox.ui.watch.runtime.snapshots import (
    _build_snapshot,
    _next_boost_lamp_level,
    _policy_auxiliary_state_predictions,
    _policy_auxiliary_state_targets,
    _publish_step_snapshots,
)
from rl_fzerox.ui.watch.runtime.telemetry import _read_live_telemetry
from rl_fzerox.ui.watch.runtime.timing import (
    RateMeter,
    _adjust_control_fps,
    _resolve_control_fps,
    _target_seconds,
)
from rl_fzerox.ui.watch.runtime.x_cup import materialize_x_cup_watch_baseline

if TYPE_CHECKING:
    from fzerox_emulator import FZeroXTelemetry
    from rl_fzerox.core.envs.observations import ObservationValue
    from rl_fzerox.core.policy.auxiliary_state import AuxiliaryStateTargetName
    from rl_fzerox.core.training.inference import PolicyRunner
    from rl_fzerox.ui.watch.runtime.cnn import _CnnActivationRunner


def _refresh_paused_cnn_activations(
    *,
    current_activations: CnnActivationSnapshot | None,
    cnn_sampler: CnnActivationSampler,
    cnn_visualization_enabled: bool,
    previous_cnn_visualization_enabled: bool,
    cnn_normalization: CnnActivationNormalizationMode,
    previous_cnn_normalization: CnnActivationNormalizationMode,
    policy_runner: _CnnActivationRunner | None,
    observation: ObservationValue,
) -> tuple[CnnActivationSnapshot | None, bool]:
    next_activations = cnn_sampler.capture(
        enabled=cnn_visualization_enabled,
        policy_runner=policy_runner,
        observation=observation,
        normalization=cnn_normalization,
        force_refresh=(
            cnn_visualization_enabled != previous_cnn_visualization_enabled
            or cnn_normalization != previous_cnn_normalization
        ),
    )
    return next_activations, next_activations is not current_activations


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
    seed_process(config.seed)
    x_cup_info = materialize_x_cup_watch_baseline(config) if config.watch.x_cup.enabled else None
    emulator = Emulator(
        core_path=config.emulator.core_path,
        rom_path=config.emulator.rom_path,
        runtime_dir=config.emulator.runtime_dir,
        baseline_state_path=config.emulator.baseline_state_path,
        renderer=config.emulator.renderer,
    )
    env = FZeroXEnv(
        backend=emulator,
        config=config.env,
        reward_config=config.reward,
        curriculum_config=config.curriculum,
    )
    env.set_sequential_track_sampling(True)
    try:
        policy_runner = _load_policy_runner(
            config.watch.policy_run_dir,
            artifact=config.watch.policy_artifact,
            device=config.watch.device,
            algorithm=config.watch.policy_algorithm,
        )
        _sync_policy_curriculum_stage(policy_runner, env)
        native_control_fps = env.backend.native_fps / config.env.action_repeat
        target_control_fps = _resolve_control_fps(
            config.watch.control_fps,
            native_control_fps=native_control_fps,
        )
        target_control_seconds = _target_seconds(target_control_fps)
        control_rate = RateMeter(window=60)
        last_logged_reload_error: str | None = None
        episode = 0
        best_finish_position: int | None = None
        best_finish_times: dict[str, int] = {}
        latest_finish_times: dict[str, int] = {}
        latest_finish_deltas_ms: dict[str, int] = {}
        failed_track_attempts: frozenset[str] = frozenset()
        paused = False
        deterministic_policy = bool(config.watch.deterministic_policy)
        manual_control_enabled = policy_runner is None
        manual_control_state = ControllerState()
        committed_policy_action: ActionValue | None = None
        committed_action_mask_branches = env.action_mask_branches()
        cnn_visualization_enabled = False
        auxiliary_visualization_enabled = False
        cnn_normalization = DEFAULT_CNN_ACTIVATION_NORMALIZATION
        cnn_sampler = CnnActivationSampler(refresh_interval_steps=1)
        persistent_locked_reset_course_id: str | None = None
        one_shot_reset_course_id: str | None = None
        sequential_course_ids = _watch_sequential_course_ids(config.env.track_sampling.entries)
        watch_zeroed_state_features = configured_watch_zeroed_features(config)
        auxiliary_target_names: tuple[AuxiliaryStateTargetName, ...] = (
            ()
            if config.policy is None
            else tuple(loss.name for loss in config.policy.auxiliary_state.losses)
        )

        while config.watch.episodes is None or episode < config.watch.episodes:
            effective_reset_course_id = (
                one_shot_reset_course_id
                if one_shot_reset_course_id is not None
                else persistent_locked_reset_course_id
            )
            env.set_locked_reset_course(effective_reset_course_id)
            reset_seed = config.seed if episode == 0 else None
            raw_observation, raw_info = env.reset(seed=reset_seed)
            one_shot_reset_course_id = None
            if x_cup_info is not None:
                raw_info.update(x_cup_info)
            observation, info = apply_watch_state_feature_zeroing(
                raw_observation,
                raw_info,
                watch_zeroed_features=watch_zeroed_state_features,
            )
            _reset_policy_runner(policy_runner)
            reset_info = dict(info)
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
                    cnn_activations=None,
                    best_finish_position=best_finish_position,
                    best_finish_times=best_finish_times,
                    latest_finish_times=latest_finish_times,
                    latest_finish_deltas_ms=latest_finish_deltas_ms,
                    failed_track_attempts=failed_track_attempts,
                ),
            )

            while not (terminated or truncated):
                previous_cnn_visualization_enabled = cnn_visualization_enabled
                previous_auxiliary_visualization_enabled = auxiliary_visualization_enabled
                previous_cnn_normalization = cnn_normalization
                commands, paused, manual_control_state = drain_worker_commands(
                    command_queue,
                    paused=paused,
                    control_state=manual_control_state,
                    manual_control_enabled=manual_control_enabled,
                    cnn_visualization_enabled=cnn_visualization_enabled,
                    auxiliary_visualization_enabled=auxiliary_visualization_enabled,
                    cnn_normalization=cnn_normalization,
                )
                manual_control_enabled = (
                    True if policy_runner is None else commands.manual_control_enabled
                )
                cnn_visualization_enabled = commands.cnn_visualization_enabled
                auxiliary_visualization_enabled = (
                    bool(auxiliary_target_names) and commands.auxiliary_visualization_enabled
                )
                cnn_normalization = commands.cnn_normalization
                lock_state_changed = False
                if commands.quit_requested:
                    return
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
                            best_finish_position=best_finish_position,
                            best_finish_times=best_finish_times,
                            latest_finish_times=latest_finish_times,
                            latest_finish_deltas_ms=latest_finish_deltas_ms,
                            failed_track_attempts=failed_track_attempts,
                        ),
                    )
                if commands.toggle_track_course_lock_id is not None:
                    persistent_locked_reset_course_id = _toggle_locked_reset_course(
                        env,
                        locked_reset_course_id=persistent_locked_reset_course_id,
                        requested_course_id=commands.toggle_track_course_lock_id,
                    )
                    one_shot_reset_course_id = None
                    _sync_locked_course_info(
                        info=info,
                        reset_info=reset_info,
                        locked_reset_course_id=persistent_locked_reset_course_id,
                    )
                    break
                if commands.toggle_current_course_lock:
                    current_course_id = _current_watch_course_id(info)
                    if current_course_id is not None:
                        persistent_locked_reset_course_id = _toggle_locked_reset_course(
                            env,
                            locked_reset_course_id=persistent_locked_reset_course_id,
                            requested_course_id=current_course_id,
                        )
                        one_shot_reset_course_id = None
                        _sync_locked_course_info(
                            info=info,
                            reset_info=reset_info,
                            locked_reset_course_id=persistent_locked_reset_course_id,
                        )
                        lock_state_changed = True
                if commands.reset_mode == "current":
                    one_shot_reset_course_id = _current_watch_course_id(info)
                    break
                if commands.reset_mode == "previous":
                    persistent_locked_reset_course_id, one_shot_reset_course_id = (
                        _advance_watch_reset_course(
                            locked_reset_course_id=persistent_locked_reset_course_id,
                            current_course_id=_current_watch_course_id(info),
                            ordered_course_ids=sequential_course_ids,
                            offset=-1,
                        )
                    )
                    break
                if commands.reset_mode == "next":
                    persistent_locked_reset_course_id, one_shot_reset_course_id = (
                        _advance_watch_reset_course(
                            locked_reset_course_id=persistent_locked_reset_course_id,
                            current_course_id=_current_watch_course_id(info),
                            ordered_course_ids=sequential_course_ids,
                            offset=1,
                        )
                    )
                    break
                if lock_state_changed:
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
                            best_finish_position=best_finish_position,
                            best_finish_times=best_finish_times,
                            latest_finish_times=latest_finish_times,
                            latest_finish_deltas_ms=latest_finish_deltas_ms,
                            failed_track_attempts=failed_track_attempts,
                        ),
                    )
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
                                best_finish_position=best_finish_position,
                                best_finish_times=best_finish_times,
                                latest_finish_times=latest_finish_times,
                                latest_finish_deltas_ms=latest_finish_deltas_ms,
                                failed_track_attempts=failed_track_attempts,
                            ),
                        )
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
                            current_control_state
                        )
                        display_frames = (env.render(),)
                    else:
                        watch_step = env.step_control_watch(current_control_state)
                        observation, reward, terminated, truncated, info = watch_step.gym_result()
                        display_frames = watch_step.display_frames
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
                    current_control_state = env.last_requested_control_state
                    current_gas_level = env.last_gas_level
                if manual_control_enabled:
                    raw_observation = observation
                    raw_info = info
                if x_cup_info is not None:
                    raw_info.update(x_cup_info)
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
                best_finish_position = _update_best_finish_position(
                    best_finish_position,
                    info,
                    None,
                )
                latest_finish_deltas_ms = _update_latest_finish_deltas_ms(
                    latest_finish_deltas_ms,
                    best_finish_times,
                    info,
                    live_telemetry,
                )
                best_finish_times = _update_best_finish_times(
                    best_finish_times,
                    info,
                    live_telemetry,
                )
                latest_finish_times = _update_latest_finish_times(
                    latest_finish_times,
                    info,
                    live_telemetry,
                )
                failed_track_attempts = _update_failed_track_attempts(
                    failed_track_attempts,
                    info,
                    episode_done=terminated or truncated,
                )
                _publish_step_snapshots(
                    config=config,
                    env=env,
                    emulator=emulator,
                    snapshot_queue=snapshot_queue,
                    display_frames=display_frames,
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
                    best_finish_position=best_finish_position,
                    best_finish_times=best_finish_times,
                    latest_finish_times=latest_finish_times,
                    latest_finish_deltas_ms=latest_finish_deltas_ms,
                    failed_track_attempts=failed_track_attempts,
                    manual_control_enabled=manual_control_enabled,
                )
                committed_policy_action = current_policy_action
                committed_action_mask_branches = final_action_mask_branches
                current_telemetry = live_telemetry
                current_auxiliary_predictions = final_auxiliary_predictions
                current_auxiliary_targets = final_auxiliary_targets
                if target_control_seconds is not None:
                    now = time.perf_counter()
                    next_step_time = max(next_step_time + target_control_seconds, now)
            episode += 1
    finally:
        env.close()


def _toggle_locked_reset_course(
    env: FZeroXEnv,
    *,
    locked_reset_course_id: str | None,
    requested_course_id: str,
) -> str | None:
    next_course_id = None if requested_course_id == locked_reset_course_id else requested_course_id
    env.set_locked_reset_course(next_course_id)
    return next_course_id


def _current_watch_course_id(info: dict[str, object]) -> str | None:
    course_id = info.get("track_course_id")
    return course_id if isinstance(course_id, str) and course_id else None


def _watch_sequential_course_ids(
    entries: tuple[TrackSamplingEntryConfig, ...],
) -> tuple[str, ...]:
    seen_course_keys: set[str] = set()
    ordered_course_ids: list[str] = []
    for entry in entries:
        course_key = _watch_entry_course_key(entry)
        if course_key in seen_course_keys:
            continue
        seen_course_keys.add(course_key)
        if entry.course_id:
            ordered_course_ids.append(entry.course_id)
    return tuple(ordered_course_ids)


def _watch_entry_course_key(entry: TrackSamplingEntryConfig) -> str:
    if entry.course_id:
        return f"course_id:{entry.course_id}"
    if entry.course_ref:
        return f"course_ref:{entry.course_ref}"
    if entry.course_index is not None:
        return f"course_index:{int(entry.course_index)}"
    return f"entry:{entry.id}"


def _current_auxiliary_predictions(
    *,
    policy_runner: PolicyRunner | None,
    enabled: bool,
    observation: ObservationValue,
    target_names: tuple[AuxiliaryStateTargetName, ...],
) -> dict[str, object] | None:
    if not enabled:
        return None
    return _policy_auxiliary_state_predictions(
        policy_runner=policy_runner,
        observation=observation,
        target_names=target_names,
    )


def _current_auxiliary_targets(
    *,
    telemetry: FZeroXTelemetry | None,
    enabled: bool,
    target_names: tuple[AuxiliaryStateTargetName, ...],
) -> dict[str, object] | None:
    if not enabled:
        return None
    if telemetry is None:
        return {}
    return _policy_auxiliary_state_targets(telemetry, target_names=target_names)


def _adjacent_watch_course_id(
    *,
    current_course_id: str | None,
    ordered_course_ids: tuple[str, ...],
    offset: int,
) -> str | None:
    if current_course_id is None:
        return None
    if not ordered_course_ids:
        return current_course_id
    try:
        current_index = ordered_course_ids.index(current_course_id)
    except ValueError:
        return current_course_id
    next_index = (current_index + offset) % len(ordered_course_ids)
    return ordered_course_ids[next_index]


def _advance_watch_reset_course(
    *,
    locked_reset_course_id: str | None,
    current_course_id: str | None,
    ordered_course_ids: tuple[str, ...],
    offset: int,
) -> tuple[str | None, str | None]:
    base_course_id = locked_reset_course_id or current_course_id
    next_course_id = _adjacent_watch_course_id(
        current_course_id=base_course_id,
        ordered_course_ids=ordered_course_ids,
        offset=offset,
    )
    if locked_reset_course_id is None:
        return None, next_course_id
    return next_course_id, None


def _sync_locked_course_info(
    *,
    info: dict[str, object],
    reset_info: dict[str, object],
    locked_reset_course_id: str | None,
) -> None:
    if locked_reset_course_id is None:
        info.pop("track_sampling_locked_course_id", None)
        reset_info.pop("track_sampling_locked_course_id", None)
        return
    info["track_sampling_locked_course_id"] = locked_reset_course_id
    reset_info["track_sampling_locked_course_id"] = locked_reset_course_id
