# src/rl_fzerox/ui/watch/view/screen/render.py
from __future__ import annotations

from rl_fzerox.core.config.schema import (
    ActionRuntimeConfig,
    CurriculumStageConfig,
    TrackConfig,
    TrackSamplingConfig,
    TrackSamplingEntryConfig,
    WatchAppConfig,
)
from rl_fzerox.core.domain.action_adapters import CONTINUOUS_DRIVE_ACTION_ADAPTERS
from rl_fzerox.core.envs import observations as observation_access
from rl_fzerox.core.envs.telemetry import telemetry_boost_active
from rl_fzerox.ui.watch.runtime.ipc import WatchSnapshot
from rl_fzerox.ui.watch.runtime.telemetry import _telemetry_from_data
from rl_fzerox.ui.watch.runtime.timing import RateMeter
from rl_fzerox.ui.watch.view.screen.frame import FrameRenderData, _draw_frame
from rl_fzerox.ui.watch.view.screen.types import (
    PygameModule,
    PygameSurface,
    ViewerFonts,
    ViewerHitboxes,
)


def draw_watch_frame(
    *,
    pygame: PygameModule,
    screen: PygameSurface,
    fonts: ViewerFonts,
    config: WatchAppConfig,
    snapshot: WatchSnapshot,
    paused: bool,
    render_rate: RateMeter,
    target_render_fps: float | None,
    panel_tab_index: int = 0,
    record_tab_index: int = 0,
) -> ViewerHitboxes:
    """Render one worker state packet without leaking env/policy logic into drawing."""

    action_config = config.env.action.runtime()
    telemetry = _telemetry_from_data(snapshot.telemetry_data)
    draw_info = _with_viewer_rates(
        snapshot.info,
        native_fps=snapshot.native_fps,
        action_repeat=config.env.action_repeat,
        current_control_fps=snapshot.control_fps,
        current_render_fps=render_rate.rate_hz(),
        target_control_fps=snapshot.target_control_fps,
        target_render_fps=target_render_fps,
    )
    track_pool_records = _track_pool_records(
        config,
        draw_info,
        policy_stage_name=snapshot.policy_curriculum_stage,
    )
    _add_config_track_info(draw_info, config, track_pool_records=track_pool_records)
    return _draw_frame(
        pygame=pygame,
        screen=screen,
        fonts=fonts,
        data=FrameRenderData(
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
            thrust_deadzone_threshold=_thrust_deadzone_threshold(action_config),
            thrust_full_threshold=_thrust_full_threshold(action_config),
            boost_active=telemetry_boost_active(telemetry),
            boost_lamp_level=snapshot.boost_lamp_level,
            action_mask_branches=snapshot.action_mask_branches,
            policy_label=snapshot.policy_label,
            policy_curriculum_stage=snapshot.policy_curriculum_stage,
            policy_num_timesteps=snapshot.policy_num_timesteps,
            policy_deterministic=snapshot.policy_deterministic,
            manual_control_enabled=snapshot.manual_control_enabled,
            policy_action=snapshot.policy_action,
            policy_reload_age_seconds=snapshot.policy_reload_age_seconds,
            policy_reload_error=snapshot.policy_reload_error,
            cnn_activations=snapshot.cnn_activations,
            best_finish_position=snapshot.best_finish_position,
            best_finish_times=snapshot.best_finish_times,
            latest_finish_times=snapshot.latest_finish_times,
            latest_finish_deltas_ms=snapshot.latest_finish_deltas_ms,
            failed_track_attempts=snapshot.failed_track_attempts,
            track_pool_records=track_pool_records,
            panel_tab_index=panel_tab_index,
            record_tab_index=record_tab_index,
            continuous_drive_deadzone=action_config.continuous_drive_deadzone,
            continuous_air_brake_mode=action_config.continuous_air_brake_mode,
            continuous_air_brake_disabled=snapshot.continuous_air_brake_disabled,
            action_repeat=config.env.action_repeat,
            max_episode_steps=config.env.max_episode_steps,
            progress_frontier_stall_limit_frames=config.env.progress_frontier_stall_limit_frames,
            stuck_min_speed_kph=config.env.stuck_min_speed_kph,
            telemetry=telemetry,
            train_config=config.train,
            policy_config=config.policy,
        ),
    )


def _thrust_warning_threshold(config: WatchAppConfig) -> float | None:
    if config.reward.gas_underuse_penalty >= 0.0:
        return None
    return float(config.reward.gas_underuse_threshold)


def _thrust_deadzone_threshold(action_config: ActionRuntimeConfig) -> float | None:
    if action_config.name not in CONTINUOUS_DRIVE_ACTION_ADAPTERS:
        return None
    return float(action_config.continuous_drive_deadzone)


def _thrust_full_threshold(action_config: ActionRuntimeConfig) -> float | None:
    if action_config.name not in CONTINUOUS_DRIVE_ACTION_ADAPTERS:
        return None
    return float(action_config.continuous_drive_full_threshold)


def _add_config_track_info(
    info: dict[str, object],
    config: WatchAppConfig,
    *,
    track_pool_records: tuple[dict[str, object], ...] | None = None,
) -> None:
    records = (
        _track_pool_records(config, info)
        if track_pool_records is None
        else track_pool_records
    )
    registry_match = _track_record_matching_info(info, records)
    if registry_match:
        for key, value in registry_match.items():
            info.setdefault(key, value)
    has_runtime_track = bool(info.get("track_display_name") or info.get("track_id"))
    if has_runtime_track:
        return
    info.update(_track_config_record(config.track))


def _track_pool_records(
    config: WatchAppConfig,
    info: dict[str, object] | None = None,
    *,
    policy_stage_name: str | None = None,
) -> tuple[dict[str, object], ...]:
    track_sampling = _active_track_sampling(
        config,
        info,
        policy_stage_name=policy_stage_name,
    )
    if track_sampling.enabled and track_sampling.entries:
        return tuple(_track_sampling_record(entry) for entry in track_sampling.entries)
    track_record = _track_config_record(config.track)
    return (track_record,) if track_record else ()


def _active_track_sampling(
    config: WatchAppConfig,
    info: dict[str, object] | None,
    *,
    policy_stage_name: str | None,
) -> TrackSamplingConfig:
    stage = _active_curriculum_stage(
        config,
        info,
        policy_stage_name=policy_stage_name,
    )
    if stage is None or stage.track_sampling is None:
        return config.env.track_sampling
    return stage.track_sampling


def _active_curriculum_stage(
    config: WatchAppConfig,
    info: dict[str, object] | None,
    *,
    policy_stage_name: str | None,
) -> CurriculumStageConfig | None:
    if not config.curriculum.enabled:
        return None

    stage_index = _curriculum_stage_index(info)
    if stage_index is not None and 0 <= stage_index < len(config.curriculum.stages):
        return config.curriculum.stages[stage_index]

    stage_name = _curriculum_stage_name(info) or policy_stage_name
    if stage_name is None:
        return None
    for stage in config.curriculum.stages:
        if stage.name == stage_name:
            return stage
    return None


def _curriculum_stage_index(info: dict[str, object] | None) -> int | None:
    if info is None:
        return None
    value = info.get("curriculum_stage")
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _curriculum_stage_name(info: dict[str, object] | None) -> str | None:
    if info is None:
        return None
    value = info.get("curriculum_stage_name")
    if not isinstance(value, str) or not value:
        return None
    return value


def _track_sampling_record(entry: TrackSamplingEntryConfig) -> dict[str, object]:
    info: dict[str, object] = {
        "track_id": entry.id,
        "track_baseline_state_path": str(entry.baseline_state_path),
        "track_sampling_weight": float(entry.weight),
    }
    if entry.display_name is not None:
        info["track_display_name"] = entry.display_name
    if entry.course_ref is not None:
        info["track_course_ref"] = entry.course_ref
    if entry.course_id is not None:
        info["track_course_id"] = entry.course_id
    if entry.course_name is not None:
        info["track_course_name"] = entry.course_name
    if entry.course_index is not None:
        info["track_course_index"] = int(entry.course_index)
    if entry.mode is not None:
        info["track_mode"] = entry.mode
    if entry.vehicle is not None:
        info["track_vehicle"] = entry.vehicle
    if entry.vehicle_name is not None:
        info["track_vehicle_name"] = entry.vehicle_name
    if entry.engine_setting is not None:
        info["track_engine_setting"] = entry.engine_setting
    if entry.records is not None:
        info.update(entry.records.info())
    return info


def _track_config_record(track: TrackConfig) -> dict[str, object]:
    info: dict[str, object] = {}
    if track.display_name is not None:
        info["track_display_name"] = track.display_name
    if track.id is not None:
        info["track_id"] = track.id
    if track.course_ref is not None:
        info["track_course_ref"] = track.course_ref
    if track.course_id is not None:
        info["track_course_id"] = track.course_id
    if track.course_name is not None:
        info["track_course_name"] = track.course_name
    if track.course_index is not None:
        info["track_course_index"] = int(track.course_index)
    if track.mode is not None:
        info["track_mode"] = track.mode
    if track.vehicle is not None:
        info["track_vehicle"] = track.vehicle
    if track.vehicle_name is not None:
        info["track_vehicle_name"] = track.vehicle_name
    if track.engine_setting is not None:
        info["track_engine_setting"] = track.engine_setting
    if track.baseline_state_path is not None:
        info["track_baseline_state_path"] = str(track.baseline_state_path)
    if track.records is not None:
        info.update(track.records.info())
    return info


def _track_record_matching_info(
    info: dict[str, object],
    track_pool_records: tuple[dict[str, object], ...],
) -> dict[str, object]:
    track_id = info.get("track_id")
    if isinstance(track_id, str) and track_id:
        for record in track_pool_records:
            if record.get("track_id") == track_id:
                return record

    course_index = info.get("track_course_index", info.get("course_index"))
    if isinstance(course_index, bool) or not isinstance(course_index, int):
        return {}
    for record in track_pool_records:
        if record.get("track_course_index") == course_index:
            return record
    return {}


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
    return observation_access.state_feature_names(
        config.env.observation.state_profile,
        course_context=config.env.observation.course_context,
        ground_effect_context=config.env.observation.ground_effect_context,
        action_history_len=config.env.observation.action_history_len,
        action_history_controls=config.env.observation.action_history_controls,
        state_components=config.env.observation.state_components_data(),
    )


def _with_viewer_rates(
    info: dict[str, object],
    *,
    native_fps: float,
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
    draw_info["native_fps"] = native_fps
    draw_info["control_fps_target"] = (
        "unlimited" if target_control_fps is None else target_control_fps
    )
    draw_info["game_fps_target"] = (
        "unlimited" if target_control_fps is None else target_control_fps * float(action_repeat)
    )
    draw_info["render_fps_target"] = "unlimited" if target_render_fps is None else target_render_fps
    return draw_info
