# src/rl_fzerox/ui/watch/app.py
from __future__ import annotations

import time

from fzerox_emulator import ControllerState, Emulator, FZeroXTelemetry
from rl_fzerox.core.config.schema import WatchAppConfig
from rl_fzerox.core.envs import FZeroXEnv
from rl_fzerox.core.envs import observations as observation_utils
from rl_fzerox.core.envs.actions import ActionValue
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
    _policy_curriculum_stage,
    _policy_deterministic,
    _policy_label,
    _policy_reload_age_seconds,
    _policy_reload_error,
    _read_live_telemetry,
    _reset_policy_runner,
    _save_baseline_state,
    _sync_policy_curriculum_stage,
    _with_viewer_fps,
)

__all__ = ["run_viewer"]

_CONTROL_FPS_ADJUST_STEP = 15.0
_MIN_CONTROL_FPS = 1.0
_PAUSED_EVENT_POLL_SECONDS = 0.01


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
    env = FZeroXEnv(
        backend=emulator,
        config=config.env,
        reward_config=config.reward,
        curriculum_config=config.curriculum,
    )
    pygame.init()

    try:
        policy_runner = _load_policy_runner(
            config.watch.policy_run_dir,
            artifact=config.watch.policy_artifact,
            device=config.watch.device,
        )
        _sync_policy_curriculum_stage(policy_runner, env)
        screen = None
        fonts = None
        paused = False
        native_control_fps: float | None = None
        target_control_fps: float | None = None
        target_render_fps: float | None = None
        target_control_seconds: float | None = None
        target_render_seconds: float | None = None
        last_control_time: float | None = None
        last_draw_time: float | None = None
        viewer_fps = 0.0
        control_fps = 0.0
        last_logged_reload_error: str | None = None
        episode = 0
        best_finish_position: int | None = None
        while config.watch.episodes is None or episode < config.watch.episodes:
            reset_seed = config.seed if episode == 0 else None
            observation, info = env.reset(seed=reset_seed)
            _reset_policy_runner(policy_runner)
            observation_image = observation_utils.observation_image(observation)
            observation_state = observation_utils.observation_state(observation)
            raw_frame = env.render()
            reset_info = dict(info)
            current_control_state = ControllerState()
            current_policy_action: ActionValue | None = None
            telemetry = _read_live_telemetry(emulator)
            last_control_time = None

            if screen is None or fonts is None:
                screen = _create_screen(
                    pygame,
                    (raw_frame.shape[1], raw_frame.shape[0]),
                    observation_image.shape,
                )
                fonts = _create_fonts(pygame)
                native_control_fps = env.backend.native_fps / config.env.action_repeat
                target_control_fps = _resolve_control_fps(
                    config.watch.control_fps,
                    native_control_fps=native_control_fps,
                )
                target_render_fps = _resolve_render_fps(
                    config.watch.render_fps,
                    native_fps=env.backend.native_fps,
                )
                target_control_seconds = _target_seconds(target_control_fps)
                target_render_seconds = _target_seconds(target_render_fps)

            viewer_input = _poll_viewer_input(pygame)
            if viewer_input.control_fps_delta:
                target_control_fps = _adjust_control_fps(
                    target_control_fps,
                    viewer_input.control_fps_delta,
                    native_control_fps=native_control_fps,
                )
                target_control_seconds = _target_seconds(target_control_fps)
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
                observation_image.shape,
            )

            draw_info, last_draw_time, viewer_fps = _with_viewer_fps(
                info,
                last_draw_time=last_draw_time,
                current_viewer_fps=viewer_fps,
                action_repeat=config.env.action_repeat,
                current_control_fps=control_fps,
                target_control_fps=target_control_fps,
                target_render_fps=target_render_fps,
            )
            _draw_frame(
                pygame=pygame,
                screen=screen,
                fonts=fonts,
                raw_frame=raw_frame,
                observation=observation_image,
                observation_state=observation_state,
                observation_state_feature_names=_observation_state_feature_names(config, info),
                episode=episode,
                info=draw_info,
                reset_info=reset_info,
                episode_reward=0.0,
                paused=paused,
                control_state=current_control_state,
                policy_label=_policy_label(policy_runner),
                policy_curriculum_stage=_policy_curriculum_stage(policy_runner),
                policy_deterministic=_policy_deterministic(
                    policy_runner,
                    config.watch.deterministic_policy,
                ),
                policy_action=current_policy_action,
                policy_reload_age_seconds=_policy_reload_age_seconds(policy_runner),
                policy_reload_error=policy_reload_error,
                best_finish_position=best_finish_position,
                continuous_drive_mode=config.env.action.continuous_drive_mode,
                continuous_drive_deadzone=config.env.action.continuous_drive_deadzone,
                continuous_air_brake_mode=config.env.action.continuous_air_brake_mode,
                continuous_air_brake_disabled=_continuous_air_brake_disabled(
                    config,
                    telemetry,
                ),
                action_repeat=config.env.action_repeat,
                max_episode_steps=config.env.max_episode_steps,
                stuck_step_limit=config.env.stuck_step_limit,
                wrong_way_timer_limit=config.env.wrong_way_timer_limit,
                progress_frontier_stall_limit_frames=(
                    config.env.progress_frontier_stall_limit_frames
                ),
                stuck_min_speed_kph=config.env.stuck_min_speed_kph,
                telemetry=telemetry,
            )

            terminated = False
            truncated = False
            episode_reward = 0.0

            while not (terminated or truncated):
                viewer_input = _poll_viewer_input(pygame)
                if viewer_input.control_fps_delta:
                    target_control_fps = _adjust_control_fps(
                        target_control_fps,
                        viewer_input.control_fps_delta,
                        native_control_fps=native_control_fps,
                    )
                    target_control_seconds = _target_seconds(target_control_fps)
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
                    if _should_draw(last_draw_time, target_render_seconds):
                        observation_image = observation_utils.observation_image(observation)
                        observation_state = observation_utils.observation_state(observation)
                        raw_frame = env.render()
                        telemetry = _read_live_telemetry(emulator)
                        best_finish_position = _update_best_finish_position(
                            best_finish_position,
                            info,
                            telemetry,
                        )
                        draw_info, last_draw_time, viewer_fps = _with_viewer_fps(
                            info,
                            last_draw_time=last_draw_time,
                            current_viewer_fps=viewer_fps,
                            action_repeat=config.env.action_repeat,
                            current_control_fps=control_fps,
                            target_control_fps=target_control_fps,
                            target_render_fps=target_render_fps,
                        )
                        _draw_frame(
                            pygame=pygame,
                            screen=screen,
                            fonts=fonts,
                            raw_frame=raw_frame,
                            observation=observation_image,
                            observation_state=observation_state,
                            observation_state_feature_names=_observation_state_feature_names(
                                config,
                                info,
                            ),
                            episode=episode,
                            info=draw_info,
                            reset_info=reset_info,
                            episode_reward=episode_reward,
                            paused=True,
                            control_state=current_control_state,
                            policy_label=_policy_label(policy_runner),
                            policy_curriculum_stage=_policy_curriculum_stage(policy_runner),
                            policy_deterministic=_policy_deterministic(
                                policy_runner,
                                config.watch.deterministic_policy,
                            ),
                            policy_action=current_policy_action,
                            policy_reload_age_seconds=_policy_reload_age_seconds(policy_runner),
                            policy_reload_error=policy_reload_error,
                            best_finish_position=best_finish_position,
                            continuous_drive_mode=config.env.action.continuous_drive_mode,
                            continuous_drive_deadzone=config.env.action.continuous_drive_deadzone,
                            continuous_air_brake_mode=config.env.action.continuous_air_brake_mode,
                            continuous_air_brake_disabled=_continuous_air_brake_disabled(
                                config,
                                telemetry,
                            ),
                            action_repeat=config.env.action_repeat,
                            max_episode_steps=config.env.max_episode_steps,
                            stuck_step_limit=config.env.stuck_step_limit,
                            wrong_way_timer_limit=config.env.wrong_way_timer_limit,
                            progress_frontier_stall_limit_frames=(
                                config.env.progress_frontier_stall_limit_frames
                            ),
                            stuck_min_speed_kph=config.env.stuck_min_speed_kph,
                            telemetry=telemetry,
                        )
                    time.sleep(_PAUSED_EVENT_POLL_SECONDS)
                    continue

                if paused and viewer_input.step_once:
                    if policy_runner is None:
                        observation, reward, terminated, truncated, info = env.step_frame(
                            current_control_state
                        )
                        current_policy_action = None
                    else:
                        _sync_policy_curriculum_stage(policy_runner, env)
                        action = policy_runner.predict(
                            observation,
                            deterministic=config.watch.deterministic_policy,
                            action_masks=env.action_masks(),
                        )
                        current_policy_action = action
                        current_control_state = env.action_to_control_state(action)
                        observation, reward, terminated, truncated, info = env.step_control(
                            current_control_state
                        )
                    observation_image = observation_utils.observation_image(observation)
                    observation_state = observation_utils.observation_state(observation)
                    raw_frame = env.render()
                    episode_reward += reward
                    telemetry = _read_live_telemetry(emulator)
                    best_finish_position = _update_best_finish_position(
                        best_finish_position,
                        info,
                        telemetry,
                    )
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
                        action_repeat=config.env.action_repeat,
                        current_control_fps=control_fps,
                        target_control_fps=target_control_fps,
                        target_render_fps=target_render_fps,
                    )
                    _draw_frame(
                        pygame=pygame,
                        screen=screen,
                        fonts=fonts,
                        raw_frame=raw_frame,
                        observation=observation_image,
                        observation_state=observation_state,
                        observation_state_feature_names=_observation_state_feature_names(
                            config,
                            info,
                        ),
                        episode=episode,
                        info=draw_info,
                        reset_info=reset_info,
                        episode_reward=episode_reward,
                        paused=True,
                        control_state=current_control_state,
                        policy_label=_policy_label(policy_runner),
                        policy_curriculum_stage=_policy_curriculum_stage(policy_runner),
                        policy_deterministic=_policy_deterministic(
                            policy_runner,
                            config.watch.deterministic_policy,
                        ),
                        policy_action=current_policy_action,
                        policy_reload_age_seconds=_policy_reload_age_seconds(policy_runner),
                        policy_reload_error=policy_reload_error,
                        best_finish_position=best_finish_position,
                        continuous_drive_mode=config.env.action.continuous_drive_mode,
                        continuous_drive_deadzone=config.env.action.continuous_drive_deadzone,
                        continuous_air_brake_mode=config.env.action.continuous_air_brake_mode,
                        continuous_air_brake_disabled=_continuous_air_brake_disabled(
                            config,
                            telemetry,
                        ),
                        action_repeat=config.env.action_repeat,
                        max_episode_steps=config.env.max_episode_steps,
                        stuck_step_limit=config.env.stuck_step_limit,
                        wrong_way_timer_limit=config.env.wrong_way_timer_limit,
                        progress_frontier_stall_limit_frames=(
                            config.env.progress_frontier_stall_limit_frames
                        ),
                        stuck_min_speed_kph=config.env.stuck_min_speed_kph,
                        telemetry=telemetry,
                    )
                    continue

                control_start = time.perf_counter()
                if policy_runner is None:
                    observation, reward, terminated, truncated, info = env.step_control(
                        current_control_state
                    )
                    current_policy_action = None
                else:
                    _sync_policy_curriculum_stage(policy_runner, env)
                    action = policy_runner.predict(
                        observation,
                        deterministic=config.watch.deterministic_policy,
                        action_masks=env.action_masks(),
                    )
                    current_policy_action = action
                    current_control_state = env.action_to_control_state(action)
                    observation, reward, terminated, truncated, info = env.step_control(
                        current_control_state
                    )
                control_fps, last_control_time = _with_control_fps(
                    last_control_time=last_control_time,
                    current_control_fps=control_fps,
                )
                episode_reward += reward
                best_finish_position = _update_best_finish_position(
                    best_finish_position,
                    info,
                    None,
                )
                policy_reload_error = _policy_reload_error(policy_runner)
                last_logged_reload_error = _persist_reload_error(
                    reload_error=policy_reload_error,
                    runtime_dir=config.emulator.runtime_dir,
                    last_logged_reload_error=last_logged_reload_error,
                )

                if _should_draw(last_draw_time, target_render_seconds):
                    observation_image = observation_utils.observation_image(observation)
                    observation_state = observation_utils.observation_state(observation)
                    raw_frame = env.render()
                    telemetry = _read_live_telemetry(emulator)
                    best_finish_position = _update_best_finish_position(
                        best_finish_position,
                        info,
                        telemetry,
                    )
                    screen = _ensure_screen(
                        pygame,
                        screen,
                        emulator.display_size,
                        observation_image.shape,
                    )
                    draw_info, last_draw_time, viewer_fps = _with_viewer_fps(
                        info,
                        last_draw_time=last_draw_time,
                        current_viewer_fps=viewer_fps,
                        action_repeat=config.env.action_repeat,
                        current_control_fps=control_fps,
                        target_control_fps=target_control_fps,
                        target_render_fps=target_render_fps,
                    )
                    _draw_frame(
                        pygame=pygame,
                        screen=screen,
                        fonts=fonts,
                        raw_frame=raw_frame,
                        observation=observation_image,
                        observation_state=observation_state,
                        observation_state_feature_names=_observation_state_feature_names(
                            config,
                            info,
                        ),
                        episode=episode,
                        info=draw_info,
                        reset_info=reset_info,
                        episode_reward=episode_reward,
                        paused=paused,
                        control_state=current_control_state,
                        policy_label=_policy_label(policy_runner),
                        policy_curriculum_stage=_policy_curriculum_stage(policy_runner),
                        policy_deterministic=_policy_deterministic(
                            policy_runner,
                            config.watch.deterministic_policy,
                        ),
                        policy_action=current_policy_action,
                        policy_reload_age_seconds=_policy_reload_age_seconds(policy_runner),
                        policy_reload_error=policy_reload_error,
                        best_finish_position=best_finish_position,
                        continuous_drive_mode=config.env.action.continuous_drive_mode,
                        continuous_drive_deadzone=config.env.action.continuous_drive_deadzone,
                        continuous_air_brake_mode=config.env.action.continuous_air_brake_mode,
                        continuous_air_brake_disabled=_continuous_air_brake_disabled(
                            config,
                            telemetry,
                        ),
                        action_repeat=config.env.action_repeat,
                        max_episode_steps=config.env.max_episode_steps,
                        stuck_step_limit=config.env.stuck_step_limit,
                        wrong_way_timer_limit=config.env.wrong_way_timer_limit,
                        progress_frontier_stall_limit_frames=(
                            config.env.progress_frontier_stall_limit_frames
                        ),
                        stuck_min_speed_kph=config.env.stuck_min_speed_kph,
                        telemetry=telemetry,
                    )

                _sleep_until_next_control_step(control_start, target_control_seconds)
            episode += 1
    finally:
        env.close()
        pygame.quit()


def _resolve_control_fps(
    setting: object,
    *,
    native_control_fps: float,
) -> float | None:
    """Resolve configured env-step FPS; `None` means uncapped fast-forward."""

    if setting in (None, "auto"):
        return max(native_control_fps, _MIN_CONTROL_FPS)
    if setting == "unlimited":
        return None
    if isinstance(setting, int | float):
        return max(float(setting), _MIN_CONTROL_FPS)
    raise ValueError(f"Unsupported watch control_fps value: {setting!r}")


def _resolve_render_fps(setting: object, *, native_fps: float) -> float | None:
    """Resolve render FPS cap; `None` means draw every control step."""

    if setting is None:
        return 60.0
    if setting == "auto":
        return max(native_fps, _MIN_CONTROL_FPS)
    if setting == "unlimited":
        return None
    if isinstance(setting, int | float):
        return max(float(setting), _MIN_CONTROL_FPS)
    raise ValueError(f"Unsupported watch render_fps value: {setting!r}")


def _target_seconds(target_fps: float | None) -> float | None:
    if target_fps is None:
        return None
    return 1.0 / max(target_fps, _MIN_CONTROL_FPS)


def _adjust_control_fps(
    target_control_fps: float | None,
    delta: int,
    *,
    native_control_fps: float | None,
) -> float | None:
    if target_control_fps is None and delta > 0:
        return None
    base_fps = (
        native_control_fps
        if target_control_fps is None and native_control_fps is not None
        else target_control_fps
    )
    if base_fps is None:
        base_fps = _MIN_CONTROL_FPS
    return max(_MIN_CONTROL_FPS, base_fps + (delta * _CONTROL_FPS_ADJUST_STEP))


def _with_control_fps(
    *,
    last_control_time: float | None,
    current_control_fps: float,
) -> tuple[float, float]:
    now = time.perf_counter()
    if last_control_time is None:
        return current_control_fps, now
    return _smoothed_fps(
        previous_fps=current_control_fps,
        elapsed_seconds=now - last_control_time,
    ), now


def _should_draw(last_draw_time: float | None, target_render_seconds: float | None) -> bool:
    if last_draw_time is None or target_render_seconds is None:
        return True
    return (time.perf_counter() - last_draw_time) >= target_render_seconds


def _sleep_until_next_control_step(
    control_start: float,
    target_control_seconds: float | None,
) -> None:
    if target_control_seconds is None:
        return
    delay = max(0.0, target_control_seconds - (time.perf_counter() - control_start))
    if delay:
        time.sleep(delay)


def _smoothed_fps(*, previous_fps: float, elapsed_seconds: float) -> float:
    instant_fps = 0.0 if elapsed_seconds <= 0.0 else 1.0 / elapsed_seconds
    if previous_fps <= 0.0:
        return instant_fps
    return (0.8 * previous_fps) + (0.2 * instant_fps)


def _continuous_air_brake_disabled(
    config: WatchAppConfig,
    telemetry: FZeroXTelemetry | None,
) -> bool:
    if config.env.action.continuous_air_brake_mode != "disable_on_ground":
        return False
    return telemetry is not None and not telemetry.player.airborne


def _update_best_finish_position(
    best_finish_position: int | None,
    info: dict[str, object],
    telemetry: FZeroXTelemetry | None,
) -> int | None:
    finish_position = _successful_finish_position(info, telemetry)
    if finish_position is None:
        return best_finish_position
    if best_finish_position is None:
        return finish_position
    return min(best_finish_position, finish_position)


def _successful_finish_position(
    info: dict[str, object],
    telemetry: FZeroXTelemetry | None,
) -> int | None:
    if info.get("termination_reason") != "finished":
        return None

    raw_position: object
    if telemetry is not None:
        raw_position = telemetry.player.position
    else:
        raw_position = info.get("position")
    if isinstance(raw_position, bool) or not isinstance(raw_position, int):
        return None
    if raw_position <= 0:
        return None
    return raw_position


def _observation_state_feature_names(
    config: WatchAppConfig,
    info: dict[str, object],
) -> tuple[str, ...]:
    names = info.get("observation_state_features")
    if isinstance(names, tuple) and all(isinstance(name, str) for name in names):
        return names
    if isinstance(names, list) and all(isinstance(name, str) for name in names):
        return tuple(names)
    if config.env.observation.mode != "image_state":
        return ()
    return observation_utils.state_feature_names(config.env.observation.state_profile)
