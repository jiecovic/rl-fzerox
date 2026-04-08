# src/rl_fzerox/ui/watch/app.py
from __future__ import annotations

import time

import numpy as np

from fzerox_emulator import ControllerState, Emulator
from rl_fzerox.core.config.schema import WatchAppConfig
from rl_fzerox.core.envs import FZeroXEnv
from rl_fzerox.core.seed import seed_process
from rl_fzerox.ui.watch.input import _poll_viewer_input
from rl_fzerox.ui.watch.render.frame import (
    _create_fonts,
    _create_screen,
    _draw_frame,
    _ensure_screen,
)
from rl_fzerox.ui.watch.session import (
    _load_policy_runner,
    _persist_reload_error,
    _policy_label,
    _policy_reload_age_seconds,
    _policy_reload_error,
    _save_baseline_state,
    _telemetry_from_info,
    _with_viewer_fps,
)

__all__ = ["run_viewer"]


def run_viewer(config: WatchAppConfig) -> None:
    """Run the real-time watch UI against the configured emulator."""

    try:
        import pygame
    except ImportError as exc:
        raise RuntimeError(
            "pygame is required for watching emulator output. "
            "Install with `pip install -e .[watch]`."
        ) from exc

    seed_process(config.seed)

    emulator = Emulator(
        core_path=config.emulator.core_path,
        rom_path=config.emulator.rom_path,
        runtime_dir=config.emulator.runtime_dir,
        baseline_state_path=config.emulator.baseline_state_path,
        renderer=config.emulator.renderer,
    )
    env = FZeroXEnv(backend=emulator, config=config.env)
    pygame.init()

    try:
        policy_runner = _load_policy_runner(
            config.watch.policy_run_dir,
            artifact=config.watch.policy_artifact,
        )
        screen = None
        fonts = None
        paused = False
        target_seconds: float | None = None
        last_draw_time: float | None = None
        viewer_fps = 0.0
        last_logged_reload_error: str | None = None
        episode = 0
        while config.watch.episodes is None or episode < config.watch.episodes:
            reset_seed = config.seed if episode == 0 else None
            observation, info = env.reset(seed=reset_seed)
            raw_frame = env.render()
            reset_info = dict(info)
            current_control_state = ControllerState()
            current_policy_action: np.ndarray | None = None
            telemetry = _telemetry_from_info(info)

            if screen is None or fonts is None:
                screen = _create_screen(
                    pygame,
                    (raw_frame.shape[1], raw_frame.shape[0]),
                    observation.shape,
                )
                fonts = _create_fonts(pygame)
                target_fps = config.watch.fps or (env.backend.native_fps / config.env.action_repeat)
                target_seconds = 1.0 / target_fps

            viewer_input = _poll_viewer_input(pygame)
            if viewer_input.quit_requested:
                return
            if viewer_input.toggle_pause:
                paused = not paused
            if policy_runner is None:
                current_control_state = viewer_input.control_state
            if viewer_input.save_state:
                _save_baseline_state(
                    emulator=emulator,
                    baseline_state_path=config.emulator.baseline_state_path,
                )
            policy_reload_error = _policy_reload_error(policy_runner)
            last_logged_reload_error = _persist_reload_error(
                reload_error=policy_reload_error,
                runtime_dir=config.emulator.runtime_dir,
                last_logged_reload_error=last_logged_reload_error,
            )

            screen = _ensure_screen(
                pygame,
                screen,
                emulator.display_size,
                observation.shape,
            )

            draw_info, last_draw_time, viewer_fps = _with_viewer_fps(
                info,
                last_draw_time=last_draw_time,
                current_viewer_fps=viewer_fps,
            )
            _draw_frame(
                pygame=pygame,
                screen=screen,
                fonts=fonts,
                raw_frame=raw_frame,
                observation=observation,
                episode=episode,
                info=draw_info,
                reset_info=reset_info,
                episode_reward=0.0,
                paused=paused,
                control_state=current_control_state,
                policy_label=_policy_label(policy_runner),
                policy_action=current_policy_action,
                policy_reload_age_seconds=_policy_reload_age_seconds(policy_runner),
                policy_reload_error=policy_reload_error,
                action_repeat=config.env.action_repeat,
                stuck_step_limit=config.env.stuck_step_limit,
                telemetry=telemetry,
            )

            terminated = False
            truncated = False
            episode_reward = 0.0

            while not (terminated or truncated):
                viewer_input = _poll_viewer_input(pygame)
                if viewer_input.quit_requested:
                    return
                if viewer_input.toggle_pause:
                    paused = not paused
                if policy_runner is None:
                    current_control_state = viewer_input.control_state
                if viewer_input.save_state:
                    _save_baseline_state(
                        emulator=emulator,
                        baseline_state_path=config.emulator.baseline_state_path,
                    )
                policy_reload_error = _policy_reload_error(policy_runner)
                last_logged_reload_error = _persist_reload_error(
                    reload_error=policy_reload_error,
                    runtime_dir=config.emulator.runtime_dir,
                    last_logged_reload_error=last_logged_reload_error,
                )
                if paused and not viewer_input.step_once:
                    draw_info, last_draw_time, viewer_fps = _with_viewer_fps(
                        info,
                        last_draw_time=last_draw_time,
                        current_viewer_fps=viewer_fps,
                    )
                    _draw_frame(
                        pygame=pygame,
                        screen=screen,
                        fonts=fonts,
                        raw_frame=raw_frame,
                        observation=observation,
                        episode=episode,
                        info=draw_info,
                        reset_info=reset_info,
                        episode_reward=episode_reward,
                        paused=True,
                        control_state=current_control_state,
                        policy_label=_policy_label(policy_runner),
                        policy_action=current_policy_action,
                        policy_reload_age_seconds=_policy_reload_age_seconds(policy_runner),
                        policy_reload_error=policy_reload_error,
                        action_repeat=config.env.action_repeat,
                        stuck_step_limit=config.env.stuck_step_limit,
                        telemetry=telemetry,
                    )
                    time.sleep(0.01)
                    continue

                if paused and viewer_input.step_once:
                    if policy_runner is None:
                        observation, reward, terminated, truncated, info = env.step_frame(
                            current_control_state
                        )
                        current_policy_action = None
                    else:
                        action = policy_runner.predict(observation)
                        current_policy_action = np.asarray(action, dtype=np.int64)
                        current_control_state = env.action_to_control_state(action)
                        observation, reward, terminated, truncated, info = env.step(action)
                    raw_frame = env.render()
                    episode_reward += reward
                    telemetry = _telemetry_from_info(info)
                    policy_reload_error = _policy_reload_error(policy_runner)
                    last_logged_reload_error = _persist_reload_error(
                        reload_error=policy_reload_error,
                        runtime_dir=config.emulator.runtime_dir,
                        last_logged_reload_error=last_logged_reload_error,
                    )
                    draw_info, last_draw_time, viewer_fps = _with_viewer_fps(
                        info,
                        last_draw_time=last_draw_time,
                        current_viewer_fps=viewer_fps,
                    )
                    _draw_frame(
                        pygame=pygame,
                        screen=screen,
                        fonts=fonts,
                        raw_frame=raw_frame,
                        observation=observation,
                        episode=episode,
                        info=draw_info,
                        reset_info=reset_info,
                        episode_reward=episode_reward,
                        paused=True,
                        control_state=current_control_state,
                        policy_label=_policy_label(policy_runner),
                        policy_action=current_policy_action,
                        policy_reload_age_seconds=_policy_reload_age_seconds(policy_runner),
                        policy_reload_error=policy_reload_error,
                        action_repeat=config.env.action_repeat,
                        stuck_step_limit=config.env.stuck_step_limit,
                        telemetry=telemetry,
                    )
                    continue

                frame_start = time.perf_counter()
                if policy_runner is None:
                    observation, reward, terminated, truncated, info = env.step_control(
                        current_control_state
                    )
                    current_policy_action = None
                else:
                    action = policy_runner.predict(observation)
                    current_policy_action = np.asarray(action, dtype=np.int64)
                    current_control_state = env.action_to_control_state(action)
                    observation, reward, terminated, truncated, info = env.step(action)
                raw_frame = env.render()
                episode_reward += reward
                telemetry = _telemetry_from_info(info)
                policy_reload_error = _policy_reload_error(policy_runner)
                last_logged_reload_error = _persist_reload_error(
                    reload_error=policy_reload_error,
                    runtime_dir=config.emulator.runtime_dir,
                    last_logged_reload_error=last_logged_reload_error,
                )

                screen = _ensure_screen(
                    pygame,
                    screen,
                    emulator.display_size,
                    observation.shape,
                )
                draw_info, last_draw_time, viewer_fps = _with_viewer_fps(
                    info,
                    last_draw_time=last_draw_time,
                    current_viewer_fps=viewer_fps,
                )
                _draw_frame(
                    pygame=pygame,
                    screen=screen,
                    fonts=fonts,
                    raw_frame=raw_frame,
                    observation=observation,
                    episode=episode,
                    info=draw_info,
                    reset_info=reset_info,
                    episode_reward=episode_reward,
                    paused=paused,
                    control_state=current_control_state,
                    policy_label=_policy_label(policy_runner),
                    policy_action=current_policy_action,
                    policy_reload_age_seconds=_policy_reload_age_seconds(policy_runner),
                    policy_reload_error=policy_reload_error,
                    action_repeat=config.env.action_repeat,
                    stuck_step_limit=config.env.stuck_step_limit,
                    telemetry=telemetry,
                )

                if paused:
                    continue

                if target_seconds is None:
                    raise RuntimeError("Watch target_seconds was not initialized")
                elapsed = time.perf_counter() - frame_start
                delay = max(0.0, target_seconds - elapsed)
                if delay:
                    time.sleep(delay)
            episode += 1
    finally:
        env.close()
        pygame.quit()
