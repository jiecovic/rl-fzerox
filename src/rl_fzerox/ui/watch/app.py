# src/rl_fzerox/ui/watch/app.py
from __future__ import annotations

from rl_fzerox.core.config.schema import WatchAppConfig
from rl_fzerox.core.envs import observations as observation_utils
from rl_fzerox.ui.watch.input import _poll_viewer_input
from rl_fzerox.ui.watch.render.frame import (
    _create_fonts,
    _draw_frame,
    _ensure_screen,
    _watch_game_display_size,
)
from rl_fzerox.ui.watch.runtime import (
    apply_viewer_input,
    drain_snapshot_queue,
    start_watch_worker,
    wait_initial_snapshot,
)
from rl_fzerox.ui.watch.runtime.process import WatchSnapshot
from rl_fzerox.ui.watch.runtime.telemetry import _telemetry_from_data
from rl_fzerox.ui.watch.runtime.timing import (
    RateMeter,
    _resolve_render_fps,
)
from rl_fzerox.ui.watch.session import _with_viewer_rates

__all__ = ["run_viewer"]


def run_viewer(config: WatchAppConfig) -> None:
    """Run the watch UI while a worker process advances emulator state."""

    try:
        import pygame
    except ImportError as exc:
        raise RuntimeError(
            "pygame is required for watching emulator output. "
            "Install with `pip install -e .[watch]`."
        ) from exc

    worker = start_watch_worker(config)
    pygame.init()
    render_clock = pygame.time.Clock()
    try:
        snapshot, worker_closed = wait_initial_snapshot(worker)
        target_render_fps = _resolve_render_fps(
            config.watch.render_fps,
            native_fps=snapshot.native_fps,
        )
        render_rate = RateMeter(window=60)
        game_display_size = _watch_game_display_size()
        screen = None
        fonts = _create_fonts(pygame)
        paused = False

        while True:
            render_limit = 0 if target_render_fps is None else max(1, int(target_render_fps))
            render_clock.tick(render_limit)

            viewer_input = _poll_viewer_input(pygame)
            paused = apply_viewer_input(
                worker.command_queue,
                viewer_input,
                paused=paused,
            )
            if viewer_input.quit_requested:
                return

            latest_snapshot, worker_closed = drain_snapshot_queue(
                worker,
                worker_closed=worker_closed,
            )
            if latest_snapshot is not None:
                snapshot = latest_snapshot
            elif worker_closed and not worker.process.is_alive():
                return
            elif not worker.process.is_alive():
                raise RuntimeError("watch simulation worker stopped unexpectedly")

            screen = _ensure_screen(
                pygame,
                screen,
                game_display_size,
                snapshot.observation_image.shape,
            )
            render_rate.tick()
            _draw_snapshot(
                pygame=pygame,
                screen=screen,
                fonts=fonts,
                config=config,
                snapshot=snapshot,
                paused=paused,
                render_rate=render_rate,
                target_render_fps=target_render_fps,
            )
    finally:
        worker.shutdown()
        pygame.quit()


def _draw_snapshot(
    *,
    pygame,
    screen,
    fonts,
    config: WatchAppConfig,
    snapshot: WatchSnapshot,
    paused: bool,
    render_rate: RateMeter,
    target_render_fps: float | None,
) -> None:
    telemetry = _telemetry_from_data(snapshot.telemetry_data)
    draw_info = _with_viewer_rates(
        snapshot.info,
        action_repeat=config.env.action_repeat,
        current_control_fps=snapshot.control_fps,
        current_render_fps=render_rate.rate_hz(),
        target_control_fps=snapshot.target_control_fps,
        target_render_fps=target_render_fps,
    )
    _draw_frame(
        pygame=pygame,
        screen=screen,
        fonts=fonts,
        raw_frame=snapshot.raw_frame,
        observation=snapshot.observation_image,
        observation_state=snapshot.observation_state,
        observation_state_feature_names=_observation_state_feature_names(
            config,
            snapshot.info,
        ),
        episode=snapshot.episode,
        info=draw_info,
        reset_info=snapshot.reset_info,
        episode_reward=snapshot.episode_reward,
        paused=paused,
        control_state=snapshot.control_state,
        policy_label=snapshot.policy_label,
        policy_curriculum_stage=snapshot.policy_curriculum_stage,
        policy_deterministic=snapshot.policy_deterministic,
        policy_action=snapshot.policy_action,
        policy_reload_age_seconds=snapshot.policy_reload_age_seconds,
        policy_reload_error=snapshot.policy_reload_error,
        best_finish_position=snapshot.best_finish_position,
        continuous_drive_mode=config.env.action.continuous_drive_mode,
        continuous_drive_deadzone=config.env.action.continuous_drive_deadzone,
        continuous_air_brake_mode=config.env.action.continuous_air_brake_mode,
        continuous_air_brake_disabled=snapshot.continuous_air_brake_disabled,
        action_repeat=config.env.action_repeat,
        max_episode_steps=config.env.max_episode_steps,
        stuck_step_limit=_display_stuck_step_limit(config),
        wrong_way_timer_limit=_display_wrong_way_timer_limit(config),
        progress_frontier_stall_limit_frames=config.env.progress_frontier_stall_limit_frames,
        stuck_min_speed_kph=config.env.stuck_min_speed_kph,
        telemetry=telemetry,
    )


def _display_wrong_way_timer_limit(config: WatchAppConfig) -> int | None:
    if not config.env.wrong_way_truncation_enabled:
        return None
    return config.env.wrong_way_timer_limit


def _display_stuck_step_limit(config: WatchAppConfig) -> int | None:
    if not config.env.stuck_truncation_enabled:
        return None
    return config.env.stuck_step_limit


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
    return observation_utils.state_feature_names(
        config.env.observation.state_profile,
        action_history_len=config.env.observation.action_history_len,
        action_history_controls=config.env.observation.action_history_controls,
    )
