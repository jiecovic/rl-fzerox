# src/rl_fzerox/ui/watch/runtime/worker.py
from __future__ import annotations

import time
from multiprocessing.queues import Queue as ProcessQueue

from fzerox_emulator import ControllerState, Emulator
from rl_fzerox.core.config.schema import WatchAppConfig
from rl_fzerox.core.envs import FZeroXEnv
from rl_fzerox.core.envs.actions import ActionValue
from rl_fzerox.core.envs.engine.masks import action_mask_violations
from rl_fzerox.core.envs.telemetry import telemetry_boost_active
from rl_fzerox.core.seed import seed_process
from rl_fzerox.ui.watch.runtime.baseline import _save_baseline_state
from rl_fzerox.ui.watch.runtime.episode import (
    _update_best_finish_position,
    _update_best_finish_times,
    _update_latest_finish_deltas_ms,
    _update_latest_finish_times,
)
from rl_fzerox.ui.watch.runtime.ipc import (
    WorkerClosed,
    WorkerError,
    drain_worker_commands,
    publish_worker_message,
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
    _publish_step_snapshots,
)
from rl_fzerox.ui.watch.runtime.telemetry import _read_live_telemetry
from rl_fzerox.ui.watch.runtime.timing import (
    RateMeter,
    _adjust_control_fps,
    _resolve_control_fps,
    _target_seconds,
)


def run_simulation_worker(
    config: WatchAppConfig,
    command_queue: ProcessQueue,
    snapshot_queue: ProcessQueue,
) -> None:
    """Run emulator and policy stepping in a process owned by the watch UI."""

    try:
        _run_simulation_loop(config, command_queue, snapshot_queue)
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
    try:
        policy_runner = _load_policy_runner(
            config.watch.policy_run_dir,
            artifact=config.watch.policy_artifact,
            device=config.watch.device,
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
        paused = False
        deterministic_policy = bool(config.watch.deterministic_policy)
        manual_control_state = ControllerState()

        while config.watch.episodes is None or episode < config.watch.episodes:
            reset_seed = config.seed if episode == 0 else None
            observation, info = env.reset(seed=reset_seed)
            _reset_policy_runner(policy_runner)
            reset_info = dict(info)
            current_control_state = env.last_requested_control_state
            current_gas_level = env.last_gas_level
            boost_lamp_level = 0.0
            current_policy_action: ActionValue | None = None
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
                    policy_action=current_policy_action,
                    policy_runner=policy_runner,
                    deterministic_policy=deterministic_policy,
                    policy_reload_error=policy_reload_error,
                    best_finish_position=best_finish_position,
                    best_finish_times=best_finish_times,
                    latest_finish_times=latest_finish_times,
                    latest_finish_deltas_ms=latest_finish_deltas_ms,
                ),
            )

            while not (terminated or truncated):
                commands, paused, manual_control_state = drain_worker_commands(
                    command_queue,
                    paused=paused,
                    control_state=manual_control_state,
                )
                if commands.quit_requested:
                    return
                if commands.reset_requested:
                    break
                if commands.control_fps_delta:
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
                if policy_runner is None:
                    current_control_state = commands.control_state

                policy_reload_error = _policy_reload_error(policy_runner)
                last_logged_reload_error = _persist_reload_error(
                    reload_error=policy_reload_error,
                    runtime_dir=config.emulator.runtime_dir,
                    last_logged_reload_error=last_logged_reload_error,
                )

                if commands.paused and commands.step_requests <= 0:
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
                decision_action_mask = env.action_mask_snapshot()
                if policy_runner is None:
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
                else:
                    _sync_policy_curriculum_stage(policy_runner, env)
                    decision_action_mask = env.action_mask_snapshot()
                    action = policy_runner.predict(
                        observation,
                        deterministic=deterministic_policy,
                        action_masks=decision_action_mask.flat,
                        refresh=False,
                    )
                    violations = action_mask_violations(decision_action_mask.branches, action)
                    if violations:
                        details = ", ".join(violations)
                        raise RuntimeError(f"Policy selected masked action values: {details}")
                    current_policy_action = action
                    watch_step = env.step_watch(action)
                    observation, reward, terminated, truncated, info = watch_step.gym_result()
                    display_frames = watch_step.display_frames
                    current_control_state = env.last_requested_control_state
                    current_gas_level = env.last_gas_level
                live_telemetry = _read_live_telemetry(emulator)
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
                    reset_info=reset_info,
                    episode=episode,
                    control_fps=control_rate.rate_hz(),
                    target_control_fps=target_control_fps,
                    target_control_seconds=target_control_seconds,
                    control_state=current_control_state,
                    gas_level=current_gas_level,
                    boost_lamp_level=boost_lamp_level,
                    action_mask_branches=decision_action_mask.branches,
                    policy_action=current_policy_action,
                    policy_runner=policy_runner,
                    deterministic_policy=deterministic_policy,
                    policy_reload_error=policy_reload_error,
                    best_finish_position=best_finish_position,
                    best_finish_times=best_finish_times,
                    latest_finish_times=latest_finish_times,
                    latest_finish_deltas_ms=latest_finish_deltas_ms,
                )
                if target_control_seconds is not None:
                    now = time.perf_counter()
                    next_step_time = max(next_step_time + target_control_seconds, now)
            episode += 1
    finally:
        env.close()
