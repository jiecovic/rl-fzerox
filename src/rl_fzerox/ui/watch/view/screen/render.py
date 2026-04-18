# src/rl_fzerox/ui/watch/view/screen/render.py
from __future__ import annotations

from rl_fzerox.core.config.schema import WatchAppConfig
from rl_fzerox.core.envs import observations as observation_utils
from rl_fzerox.core.envs.telemetry import telemetry_boost_active
from rl_fzerox.ui.watch.runtime.ipc import WatchSnapshot
from rl_fzerox.ui.watch.runtime.telemetry import _telemetry_from_data
from rl_fzerox.ui.watch.runtime.timing import RateMeter
from rl_fzerox.ui.watch.view.screen.frame import _draw_frame


def draw_watch_frame(
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
    """Render one worker state packet without leaking env/policy logic into drawing."""

    telemetry = _telemetry_from_data(snapshot.telemetry_data)
    draw_info = _with_viewer_rates(
        snapshot.info,
        action_repeat=config.env.action_repeat,
        current_control_fps=snapshot.control_fps,
        current_render_fps=render_rate.rate_hz(),
        target_control_fps=snapshot.target_control_fps,
        target_render_fps=target_render_fps,
    )
    _add_config_track_info(draw_info, config)
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
        gas_level=snapshot.gas_level,
        thrust_warning_threshold=_thrust_warning_threshold(config),
        boost_active=telemetry_boost_active(telemetry),
        boost_lamp_level=snapshot.boost_lamp_level,
        policy_label=snapshot.policy_label,
        policy_curriculum_stage=snapshot.policy_curriculum_stage,
        policy_deterministic=snapshot.policy_deterministic,
        policy_action=snapshot.policy_action,
        policy_reload_age_seconds=snapshot.policy_reload_age_seconds,
        policy_reload_error=snapshot.policy_reload_error,
        best_finish_position=snapshot.best_finish_position,
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


def _thrust_warning_threshold(config: WatchAppConfig) -> float | None:
    if config.reward.gas_underuse_penalty >= 0.0:
        return None
    return float(config.reward.gas_underuse_threshold)


def _add_config_track_info(info: dict[str, object], config: WatchAppConfig) -> None:
    if info.get("track_display_name") or info.get("track_id"):
        return
    if config.track.display_name is not None:
        info["track_display_name"] = config.track.display_name
    if config.track.id is not None:
        info["track_id"] = config.track.id
    if config.track.course_index is not None:
        info["track_course_index"] = int(config.track.course_index)


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


def _with_viewer_rates(
    info: dict[str, object],
    *,
    action_repeat: int,
    current_control_fps: float,
    current_render_fps: float,
    target_control_fps: float | None = None,
    target_render_fps: float | None = None,
) -> dict[str, object]:
    draw_info = dict(info)
    draw_info["viewer_fps"] = current_render_fps
    draw_info["render_fps"] = current_render_fps
    draw_info["control_fps"] = current_control_fps
    draw_info["game_fps"] = current_control_fps * float(action_repeat)
    draw_info["control_fps_target"] = (
        "unlimited" if target_control_fps is None else target_control_fps
    )
    draw_info["game_fps_target"] = (
        "unlimited" if target_control_fps is None else target_control_fps * float(action_repeat)
    )
    draw_info["render_fps_target"] = "unlimited" if target_render_fps is None else target_render_fps
    return draw_info
