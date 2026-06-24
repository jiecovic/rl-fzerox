# src/rl_fzerox/ui/watch/view/screen/render.py
from __future__ import annotations

import math

from rl_fzerox.core.domain.courses import BUILT_IN_COURSES, CourseInfo, CourseRecords
from rl_fzerox.core.envs import observations as observation_access
from rl_fzerox.core.envs.telemetry import telemetry_boost_active
from rl_fzerox.core.runtime_spec.schema import (
    ActionRuntimeConfig,
    CareerModeRaceSetupConfig,
    TrackConfig,
    TrackSamplingConfig,
    TrackSamplingEntryConfig,
    WatchAppConfig,
)
from rl_fzerox.core.runtime_spec.track_sampling_identity import (
    track_sampling_course_key,
    track_sampling_reset_target_key,
)
from rl_fzerox.ui.watch.live_series import EpisodeLiveSeriesSnapshot
from rl_fzerox.ui.watch.records import record_difficulty
from rl_fzerox.ui.watch.runtime.ipc import WatchSnapshot
from rl_fzerox.ui.watch.runtime.telemetry import _telemetry_from_data
from rl_fzerox.ui.watch.runtime.timing import RateMeter
from rl_fzerox.ui.watch.view.auxiliary_metrics import AuxiliaryEpisodeMetricsSnapshot
from rl_fzerox.ui.watch.view.panels.core.tabs import WATCH_PANEL_TABS, PanelTabRegistry
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
    track_pool_records: tuple[dict[str, object], ...] | None = None,
    auxiliary_episode_metrics: AuxiliaryEpisodeMetricsSnapshot | None = None,
    live_episode_series: EpisodeLiveSeriesSnapshot | None = None,
    panel_tabs: PanelTabRegistry = WATCH_PANEL_TABS,
    panel_tab_index: int = 0,
    cnn_layer_tab_index: int = 0,
    record_tab_index: int = 0,
    policy_observation_layout_shape: tuple[int, ...] | None = None,
    policy_observation_layout_info: dict[str, object] | None = None,
) -> ViewerHitboxes:
    """Render one worker state packet without leaking env/policy logic into drawing."""

    action_config = config.env.action.runtime()
    telemetry = _telemetry_from_data(snapshot.telemetry_data)
    draw_info = _with_viewer_rates(
        snapshot.info,
        native_fps=snapshot.native_fps,
        action_repeat=snapshot.action_repeat,
        current_control_fps=snapshot.control_fps,
        current_render_fps=render_rate.rate_hz(),
        target_control_fps=snapshot.target_control_fps,
        target_render_fps=target_render_fps,
    )
    if track_pool_records is None:
        track_pool_records = track_pool_records_for_watch_snapshot(
            config,
            snapshot,
            active_track_sampling=snapshot.active_track_sampling,
        )
    _add_config_track_info(draw_info, config, track_pool_records=track_pool_records)
    _add_career_mode_info(draw_info, config)
    policy_observation = snapshot.policy_observation
    observation_layout_shape = (
        policy_observation_layout_shape
        or snapshot.policy_observation_shape
        or _default_policy_observation_layout_shape()
    )
    observation_layout_info = policy_observation_layout_info or draw_info
    return _draw_frame(
        pygame=pygame,
        screen=screen,
        fonts=fonts,
        data=FrameRenderData(
            raw_frame=snapshot.raw_frame,
            policy_observation_image=(
                None if policy_observation is None else policy_observation.image
            ),
            policy_observation_shape=snapshot.policy_observation_shape,
            policy_observation_layout_shape=observation_layout_shape,
            policy_observation_layout_info=observation_layout_info,
            observation_state=None if policy_observation is None else policy_observation.state,
            observation_state_reference=(
                None if policy_observation is None else policy_observation.state_reference
            ),
            observation_state_feature_names=_observation_state_feature_names(
                config,
                snapshot.info,
            ),
            policy_auxiliary_state_predictions=snapshot.policy_auxiliary_state_predictions,
            policy_auxiliary_state_targets=snapshot.policy_auxiliary_state_targets,
            auxiliary_episode_metrics=auxiliary_episode_metrics,
            live_episode_series=live_episode_series,
            episode=snapshot.episode,
            info=draw_info,
            reset_info=snapshot.reset_info,
            episode_reward=snapshot.episode_reward,
            paused=paused,
            recording_active=config.watch.recording.enabled,
            control_state=snapshot.control_state,
            gas_level=snapshot.gas_level,
            thrust_warning_threshold=_thrust_warning_threshold(config),
            thrust_deadzone_threshold=_thrust_deadzone_threshold(action_config),
            thrust_full_threshold=_thrust_full_threshold(action_config),
            boost_active=telemetry_boost_active(telemetry),
            boost_lamp_level=snapshot.boost_lamp_level,
            action_mask_branches=snapshot.action_mask_branches,
            policy_label=snapshot.policy_label,
            policy_num_timesteps=snapshot.policy_num_timesteps,
            policy_experience_frames=snapshot.policy_experience_frames,
            policy_deterministic=snapshot.policy_deterministic,
            manual_control_enabled=snapshot.manual_control_enabled,
            policy_action=snapshot.policy_action,
            policy_reload_age_seconds=snapshot.policy_reload_age_seconds,
            policy_reload_error=snapshot.policy_reload_error,
            cnn_activations=snapshot.cnn_activations,
            track_record_book=snapshot.track_record_book,
            track_pool_records=track_pool_records,
            allow_record_course_jumps=config.watch.managed_save_game_id is None,
            panel_tab_index=panel_tab_index,
            cnn_layer_tab_index=cnn_layer_tab_index,
            record_tab_index=record_tab_index,
            panel_tabs=panel_tabs,
            continuous_drive_deadzone=action_config.continuous_drive_deadzone,
            continuous_drive_enabled=action_config.uses_continuous_drive(),
            force_full_throttle=bool(action_config.force_full_throttle),
            continuous_pitch_enabled=(
                action_config.name == "configured_hybrid"
                and "pitch" in action_config.layout_continuous_axes
            ),
            continuous_air_brake_axis_index=action_config.continuous_air_brake_axis_index(),
            continuous_air_brake_deadzone=action_config.continuous_air_brake_deadzone,
            continuous_air_brake_full_threshold=action_config.continuous_air_brake_full_threshold,
            continuous_air_brake_min_duty=action_config.continuous_air_brake_min_duty,
            continuous_air_brake_mode=action_config.continuous_air_brake_mode,
            continuous_air_brake_disabled=snapshot.continuous_air_brake_disabled,
            action_repeat=snapshot.action_repeat,
            max_episode_steps=config.env.max_episode_steps,
            progress_frontier_stall_limit_frames=config.env.progress_frontier_stall_limit_frames,
            stuck_min_speed_kph=config.env.stuck_min_speed_kph,
            telemetry=telemetry,
            emulator_renderer=config.emulator.renderer,
            watch_device=config.watch.device,
            train_config=config.train,
            policy_config=config.policy,
        ),
    )


def _default_policy_observation_layout_shape() -> tuple[int, int, int]:
    return (72, 96, 3)


def _thrust_warning_threshold(config: WatchAppConfig) -> float | None:
    del config
    return None


def _thrust_deadzone_threshold(action_config: ActionRuntimeConfig) -> float | None:
    if not action_config.uses_continuous_drive():
        return None
    return float(action_config.continuous_drive_deadzone)


def _thrust_full_threshold(action_config: ActionRuntimeConfig) -> float | None:
    if not action_config.uses_continuous_drive():
        return None
    return float(action_config.continuous_drive_full_threshold)


def track_pool_records_for_watch_snapshot(
    config: WatchAppConfig,
    snapshot: WatchSnapshot,
    *,
    active_track_sampling: TrackSamplingConfig | None = None,
) -> tuple[dict[str, object], ...]:
    """Build the track-pool view model once per track-sampling config update."""

    return _track_pool_records(
        config,
        active_track_sampling=active_track_sampling,
    )


def _add_config_track_info(
    info: dict[str, object],
    config: WatchAppConfig,
    *,
    track_pool_records: tuple[dict[str, object], ...] | None = None,
) -> None:
    records = _track_pool_records(config) if track_pool_records is None else track_pool_records
    registry_match = _track_record_matching_info(info, records)
    if registry_match:
        for key, value in registry_match.items():
            info.setdefault(key, value)
    has_runtime_track = bool(info.get("track_display_name") or info.get("track_id"))
    if has_runtime_track:
        return
    info.update(_track_config_record(config.track))


def _add_career_mode_info(info: dict[str, object], config: WatchAppConfig) -> None:
    if config.watch.managed_save_game_id is None:
        return
    if config.watch.unlock_target_label is not None:
        info.setdefault("career_mode_target_label", config.watch.unlock_target_label)
    if config.watch.save_attempt_id is not None:
        info.setdefault("career_mode_attempt_id", config.watch.save_attempt_id)


def _track_pool_records(
    config: WatchAppConfig,
    *,
    active_track_sampling: TrackSamplingConfig | None = None,
) -> tuple[dict[str, object], ...]:
    track_sampling = active_track_sampling or config.env.track_sampling
    if track_sampling.enabled and track_sampling.entries:
        return _track_sampling_records(track_sampling.entries)
    career_records = _career_mode_track_pool_records(config)
    if career_records:
        return career_records
    track_record = _track_config_record(config.track)
    return (track_record,) if track_record else ()


def _career_mode_track_pool_records(config: WatchAppConfig) -> tuple[dict[str, object], ...]:
    if config.watch.managed_save_game_id is None:
        return ()
    setup = config.watch.career_mode_race_setup
    if setup is None:
        return ()
    return tuple(
        _career_mode_course_record(course, setup=setup)
        for course in BUILT_IN_COURSES
        if course.cup == setup.cup_id
    )


def _career_mode_course_record(
    course: CourseInfo,
    *,
    setup: CareerModeRaceSetupConfig,
) -> dict[str, object]:
    info: dict[str, object] = {
        "track_entry_id": f"career:{setup.difficulty}:{course.id}",
        "track_id": course.id,
        "track_course_key": course.id,
        "track_reset_target_key": course.id,
        "track_reset_course_key": course.id,
        "track_display_name": course.display_name,
        "track_course_ref": course.ref,
        "track_course_id": course.id,
        "track_course_name": course.display_name,
        "track_course_index": int(course.course_index),
        "track_mode": "gp_race",
        "track_gp_difficulty": setup.difficulty,
        "track_vehicle": setup.vehicle_id,
        "track_vehicle_name": setup.vehicle_display_name,
        "track_engine_setting_raw_value": int(setup.engine_setting_raw_value),
    }
    if course.records is not None:
        info.update(_course_records_info(course.records))
    return info


def _course_records_info(records: CourseRecords) -> dict[str, object]:
    info: dict[str, object] = {
        "track_record_source_label": records.source_label,
        "track_record_source_url": records.source_url,
        "track_non_agg_best_time_ms": int(records.non_agg_best.time_ms),
        "track_non_agg_best_player": records.non_agg_best.player,
        "track_non_agg_best_date": records.non_agg_best.date,
        "track_non_agg_best_mode": records.non_agg_best.mode,
        "track_non_agg_worst_time_ms": int(records.non_agg_worst.time_ms),
        "track_non_agg_worst_player": records.non_agg_worst.player,
        "track_non_agg_worst_date": records.non_agg_worst.date,
        "track_non_agg_worst_mode": records.non_agg_worst.mode,
    }
    return info


def _track_sampling_records(
    entries: tuple[TrackSamplingEntryConfig, ...],
) -> tuple[dict[str, object], ...]:
    alt_counts_by_source_entry: dict[str, int] = {}
    for entry in entries:
        source_entry_id = entry.alt_baseline_source_entry_id
        if entry.alt_baseline_id is None or source_entry_id is None:
            continue
        alt_counts_by_source_entry[source_entry_id] = (
            alt_counts_by_source_entry.get(
                source_entry_id,
                0,
            )
            + 1
        )

    return tuple(
        _track_sampling_record(
            entry,
            alt_baseline_count=alt_counts_by_source_entry.get(entry.id, 0),
        )
        for entry in entries
    )


def _track_sampling_record(
    entry: TrackSamplingEntryConfig,
    *,
    alt_baseline_count: int,
) -> dict[str, object]:
    course_key = track_sampling_course_key(
        entry_id=entry.id,
        course_id=entry.course_id,
        runtime_course_key=entry.runtime_course_key,
        course_ref=entry.course_ref,
        course_index=entry.course_index,
    )
    info: dict[str, object] = {
        "track_entry_id": entry.id,
        "track_id": entry.id,
        "track_course_key": course_key,
        "track_reset_target_key": track_sampling_reset_target_key(
            entry_id=entry.id,
            course_id=entry.course_id,
            runtime_course_key=entry.runtime_course_key,
            course_ref=entry.course_ref,
            course_index=entry.course_index,
            gp_difficulty=entry.gp_difficulty,
        ),
        "track_baseline_state_path": str(entry.baseline_state_path),
        "track_sampling_weight": float(entry.weight),
        "track_alt_baseline_count": int(alt_baseline_count),
    }
    if entry.runtime_course_key is not None:
        info["track_runtime_course_key"] = entry.runtime_course_key
        info["track_reset_course_key"] = entry.runtime_course_key
    if entry.display_name is not None:
        info["track_display_name"] = entry.display_name
    if entry.course_ref is not None:
        info["track_course_ref"] = entry.course_ref
    if entry.course_id is not None:
        info["track_course_id"] = entry.course_id
        info.setdefault("track_reset_course_key", entry.course_id)
    if entry.course_name is not None:
        info["track_course_name"] = entry.course_name
    if entry.course_index is not None:
        info["track_course_index"] = int(entry.course_index)
    if entry.mode is not None:
        info["track_mode"] = entry.mode
    if entry.gp_difficulty is not None:
        info["track_gp_difficulty"] = entry.gp_difficulty
    if entry.source_gp_difficulty is not None:
        info["track_source_gp_difficulty"] = entry.source_gp_difficulty
    if entry.vehicle is not None:
        info["track_vehicle"] = entry.vehicle
    if entry.vehicle_name is not None:
        info["track_vehicle_name"] = entry.vehicle_name
    if entry.engine_setting_raw_value is not None:
        info["track_engine_setting_raw_value"] = int(entry.engine_setting_raw_value)
    if entry.alt_baseline_id is not None:
        info["track_alt_baseline_id"] = entry.alt_baseline_id
    if entry.alt_baseline_label is not None:
        info["track_alt_baseline_label"] = entry.alt_baseline_label
    if entry.alt_baseline_source_entry_id is not None:
        info["track_alt_baseline_source_entry_id"] = entry.alt_baseline_source_entry_id
    if entry.generated_course_kind is not None:
        info["track_generated_course_kind"] = entry.generated_course_kind
    if entry.generated_course_slot is not None:
        info["track_generated_course_slot"] = int(entry.generated_course_slot)
    if entry.generated_course_generation is not None:
        info["track_generated_course_generation"] = int(entry.generated_course_generation)
    if entry.generated_course_hash is not None:
        info["track_generated_course_hash"] = entry.generated_course_hash
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
        info["track_reset_course_key"] = track.course_id
    if track.course_name is not None:
        info["track_course_name"] = track.course_name
    if track.course_index is not None:
        info["track_course_index"] = int(track.course_index)
    if track.mode is not None:
        info["track_mode"] = track.mode
    if track.gp_difficulty is not None:
        info["track_gp_difficulty"] = track.gp_difficulty
    if track.source_gp_difficulty is not None:
        info["track_source_gp_difficulty"] = track.source_gp_difficulty
    if track.vehicle is not None:
        info["track_vehicle"] = track.vehicle
    if track.vehicle_name is not None:
        info["track_vehicle_name"] = track.vehicle_name
    if track.engine_setting_raw_value is not None:
        info["track_engine_setting_raw_value"] = int(track.engine_setting_raw_value)
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
        match = _matching_track_record(
            info,
            tuple(record for record in track_pool_records if record.get("track_id") == track_id),
        )
        if match is not None:
            return match

    reset_course_key = info.get("track_reset_course_key")
    if isinstance(reset_course_key, str) and reset_course_key:
        match = _matching_track_record(
            info,
            tuple(
                record
                for record in track_pool_records
                if record.get("track_reset_course_key") == reset_course_key
            ),
        )
        if match is not None:
            return match

    course_id = info.get("track_course_id")
    if isinstance(course_id, str) and course_id:
        match = _matching_track_record(
            info,
            tuple(
                record
                for record in track_pool_records
                if record.get("track_course_id") == course_id
            ),
        )
        if match is not None:
            return match

    course_index = info.get("track_course_index", info.get("course_index"))
    if isinstance(course_index, bool) or not isinstance(course_index, int):
        return {}
    match = _matching_track_record(
        info,
        tuple(
            record
            for record in track_pool_records
            if record.get("track_course_index") == course_index
        ),
    )
    if match is not None:
        return match
    return {}


def _matching_track_record(
    info: dict[str, object],
    candidates: tuple[dict[str, object], ...],
) -> dict[str, object] | None:
    if not candidates:
        return None
    difficulty = record_difficulty(info)
    if difficulty is not None:
        for record in candidates:
            if record_difficulty(record) == difficulty:
                return record
    return candidates[0]


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
    state_components = config.env.observation.state_components_data()
    if state_components is None:
        return ()
    return observation_access.state_feature_names(
        state_components=state_components,
        split_lean_history=config.env.action.runtime().split_lean_history,
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
    explicit_game_fps = _finite_number(draw_info.get("game_fps"))
    draw_info["game_fps"] = (
        explicit_game_fps
        if explicit_game_fps is not None
        else current_control_fps * float(action_repeat)
    )
    draw_info["native_fps"] = native_fps
    draw_info["control_fps_target"] = (
        "unlimited" if target_control_fps is None else target_control_fps
    )
    explicit_game_fps_target = _finite_number(draw_info.get("game_fps_target"))
    draw_info["game_fps_target"] = (
        explicit_game_fps_target
        if explicit_game_fps_target is not None
        else (
            "unlimited" if target_control_fps is None else target_control_fps * float(action_repeat)
        )
    )
    draw_info["render_fps_target"] = "unlimited" if target_render_fps is None else target_render_fps
    return draw_info


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) and number >= 0.0 else None
