# src/rl_fzerox/ui/watch/view/screen/frame.py
from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np

from fzerox_emulator import FZeroXTelemetry, RaceControlState
from fzerox_emulator.arrays import ObservationFrame, RgbFrame, StateVector
from rl_fzerox.core.envs.actions import ActionValue
from rl_fzerox.core.envs.engine.controls import ActionMaskBranches
from rl_fzerox.core.runtime_spec.schema import PolicyConfig, TrainConfig
from rl_fzerox.ui.watch.live_series import EpisodeLiveSeriesSnapshot
from rl_fzerox.ui.watch.runtime.cnn import CnnActivationSnapshot
from rl_fzerox.ui.watch.view.auxiliary_metrics import AuxiliaryEpisodeMetricsSnapshot
from rl_fzerox.ui.watch.view.components.game_view import _draw_glass_game_view
from rl_fzerox.ui.watch.view.components.observation_strip import (
    _draw_control_viz_below_game,
    _draw_observation_preview_in_rect,
    _native_observation_preview_height,
    _native_observation_preview_width,
)
from rl_fzerox.ui.watch.view.panels.core.model import (
    _observation_preview_size,
    _preview_frame,
    _window_size,
)
from rl_fzerox.ui.watch.view.panels.core.tabs import PanelTabRegistry
from rl_fzerox.ui.watch.view.panels.rendering.draw import SidePanelData, _draw_side_panel
from rl_fzerox.ui.watch.view.panels.visuals.viz import _control_viz, _control_viz_height
from rl_fzerox.ui.watch.view.screen.layout import LAYOUT
from rl_fzerox.ui.watch.view.screen.theme import FONT_SIZES, PALETTE
from rl_fzerox.ui.watch.view.screen.types import (
    PygameModule,
    PygameSurface,
    RenderFont,
    ViewerFonts,
    ViewerHitboxes,
)

_WINDOW_TITLE = "F-Zero X Watch"
_WINDOW_ICON_SIZE = 32
_WINDOW_ICON_RADIUS = 6
_WINDOW_ICON_BACKGROUND = (28, 36, 48)
_WINDOW_ICON_TEXT = (141, 189, 255)


@dataclass(frozen=True, slots=True)
class FrameRenderData:
    """Complete state snapshot needed to render one watch frame."""

    raw_frame: RgbFrame
    policy_observation_image: ObservationFrame | None
    policy_observation_shape: tuple[int, ...] | None
    policy_observation_layout_shape: tuple[int, ...]
    observation_state: StateVector | None
    observation_state_reference: StateVector | None
    observation_state_feature_names: tuple[str, ...]
    policy_auxiliary_state_predictions: dict[str, object] | None
    policy_auxiliary_state_targets: dict[str, object] | None
    auxiliary_episode_metrics: AuxiliaryEpisodeMetricsSnapshot | None
    live_episode_series: EpisodeLiveSeriesSnapshot | None
    episode: int
    info: dict[str, object]
    reset_info: dict[str, object]
    episode_reward: float
    paused: bool
    control_state: RaceControlState
    gas_level: float
    thrust_warning_threshold: float | None
    thrust_deadzone_threshold: float | None
    thrust_full_threshold: float | None
    boost_active: bool
    boost_lamp_level: float
    action_mask_branches: ActionMaskBranches
    policy_label: str | None
    policy_curriculum_stage: str | None
    policy_num_timesteps: int | None
    policy_experience_frames: int | None
    policy_deterministic: bool | None
    manual_control_enabled: bool
    policy_action: ActionValue | None
    policy_reload_age_seconds: float | None
    policy_reload_error: str | None
    cnn_activations: CnnActivationSnapshot | None
    best_finish_position: int | None
    best_finish_ranks: dict[str, int]
    best_finish_rank_times: dict[str, int]
    best_finish_rank_setups: dict[str, dict[str, str | int]]
    best_finish_times: dict[str, int]
    best_finish_time_ranks: dict[str, int]
    best_finish_time_setups: dict[str, dict[str, str | int]]
    latest_finish_times: dict[str, int]
    latest_finish_deltas_ms: dict[str, int]
    track_attempt_stats: dict[str, dict[str, int | float]]
    failed_track_attempts: frozenset[str]
    track_pool_records: tuple[dict[str, object], ...]
    panel_tab_index: int
    cnn_layer_tab_index: int
    record_tab_index: int
    panel_tabs: PanelTabRegistry
    continuous_drive_deadzone: float
    continuous_drive_enabled: bool
    force_full_throttle: bool
    continuous_pitch_enabled: bool
    continuous_air_brake_axis_index: int | None
    continuous_air_brake_deadzone: float
    continuous_air_brake_full_threshold: float
    continuous_air_brake_min_duty: float
    continuous_air_brake_mode: str
    continuous_air_brake_disabled: bool
    action_repeat: int
    max_episode_steps: int
    progress_frontier_stall_limit_frames: int | None
    stuck_min_speed_kph: float
    telemetry: FZeroXTelemetry | None
    emulator_renderer: str
    watch_device: str
    train_config: TrainConfig | None
    policy_config: PolicyConfig | None


def _create_fonts(pygame: PygameModule) -> ViewerFonts:
    record_header = pygame.font.Font(None, FONT_SIZES.small)
    record_header.set_bold(True)
    return ViewerFonts(
        title=pygame.font.Font(None, FONT_SIZES.title),
        section=pygame.font.Font(None, FONT_SIZES.section),
        record_header=record_header,
        body=_create_mono_font(pygame, FONT_SIZES.body),
        small=pygame.font.Font(None, FONT_SIZES.small),
    )


def _create_mono_font(pygame: PygameModule, size: int) -> RenderFont:
    font_path = pygame.font.match_font(
        ("dejavusansmono", "liberationmono", "consolas", "couriernew", "monospace")
    )
    if font_path is not None:
        return pygame.font.Font(font_path, size)
    return pygame.font.SysFont("monospace", size)


def _create_screen(
    pygame: PygameModule,
    game_display_size: tuple[int, int],
    observation_shape: tuple[int, ...],
    *,
    fonts: ViewerFonts,
    info: dict[str, object],
    panel_tab_index: int = 0,
) -> PygameSurface:
    _apply_window_position_hint()
    _set_window_icon(pygame)
    screen = pygame.display.set_mode(
        _watch_window_size(
            game_display_size,
            observation_shape,
            fonts=fonts,
            info=info,
            panel_tab_index=panel_tab_index,
        )
    )
    pygame.display.set_caption(_WINDOW_TITLE)
    return screen


def _set_window_icon(pygame: PygameModule) -> None:
    icon = pygame.Surface(
        (_WINDOW_ICON_SIZE, _WINDOW_ICON_SIZE),
        flags=pygame.SRCALPHA,
    )
    pygame.draw.rect(
        icon,
        _WINDOW_ICON_BACKGROUND,
        icon.get_rect(),
        border_radius=_WINDOW_ICON_RADIUS,
    )
    font = pygame.font.Font(None, 18)
    font.set_bold(True)
    text = font.render("FX", True, _WINDOW_ICON_TEXT)
    icon.blit(text, text.get_rect(center=icon.get_rect().center))
    pygame.display.set_icon(icon)


def _ensure_screen(
    pygame: PygameModule,
    screen: PygameSurface | None,
    game_display_size: tuple[int, int],
    observation_shape: tuple[int, ...],
    *,
    fonts: ViewerFonts,
    info: dict[str, object],
    panel_tab_index: int = 0,
) -> PygameSurface:
    if screen is None:
        return _create_screen(
            pygame,
            game_display_size,
            observation_shape,
            fonts=fonts,
            info=info,
            panel_tab_index=panel_tab_index,
        )
    if screen.get_size() == _watch_window_size(
        game_display_size,
        observation_shape,
        fonts=fonts,
        info=info,
        panel_tab_index=panel_tab_index,
    ):
        return screen
    return _create_screen(
        pygame,
        game_display_size,
        observation_shape,
        fonts=fonts,
        info=info,
        panel_tab_index=panel_tab_index,
    )


def _watch_game_display_size() -> tuple[int, int]:
    return LAYOUT.game_display_size


def _watch_left_column_width(
    game_display_size: tuple[int, int],
    observation_shape: tuple[int, ...],
    *,
    info: dict[str, object],
) -> int:
    native_preview_width = _native_observation_preview_width(
        observation_shape=observation_shape,
        info=info,
    )
    return max(game_display_size[0], native_preview_width + (2 * LAYOUT.preview_padding))


def _watch_window_size(
    game_display_size: tuple[int, int],
    observation_shape: tuple[int, ...],
    *,
    fonts: ViewerFonts,
    info: dict[str, object],
    panel_tab_index: int = 0,
) -> tuple[int, int]:
    left_column_width = _watch_left_column_width(
        game_display_size,
        observation_shape,
        info=info,
    )
    left_content_width = left_column_width - (2 * LAYOUT.preview_padding)
    left_column_height = game_display_size[1] + LAYOUT.preview_gap + _control_viz_height(fonts)
    left_column_height += LAYOUT.preview_gap + _native_observation_preview_height(
        fonts=fonts,
        observation_shape=observation_shape,
        info=info,
        width=left_content_width,
    )
    left_column_height += LAYOUT.preview_padding
    _, baseline_height = _window_size(
        game_display_size,
        observation_shape,
        panel_tab_index=panel_tab_index,
    )
    return (
        left_column_width + LAYOUT.preview_gap + LAYOUT.panel_width,
        max(baseline_height, left_column_height),
    )


def _draw_frame(
    *,
    pygame: PygameModule,
    screen: PygameSurface,
    fonts: ViewerFonts,
    data: FrameRenderData,
) -> ViewerHitboxes:
    game_display_size = _watch_game_display_size()
    observation_layout_shape = data.policy_observation_layout_shape
    left_column_width = _watch_left_column_width(
        game_display_size,
        observation_layout_shape,
        info=data.info,
    )
    game_surface = _rgb_surface(pygame, data.raw_frame)

    screen.fill(PALETTE.app_background)
    _draw_glass_game_view(
        pygame=pygame,
        screen=screen,
        fonts=fonts,
        surface=game_surface,
        outer_size=game_display_size,
        course_label=_game_course_overlay_label(data.info, reset_info=data.reset_info),
        speed_label=_game_speed_overlay_label(data.info, action_repeat=data.action_repeat),
    )
    control_viz = _control_viz(
        data.control_state,
        gas_level=data.gas_level,
        thrust_warning_threshold=data.thrust_warning_threshold,
        thrust_deadzone_threshold=data.thrust_deadzone_threshold,
        thrust_full_threshold=data.thrust_full_threshold,
        engine_setting_level=_engine_setting_level(data.info),
        speed_kph=_speed_kph(data.telemetry),
        energy_fraction=_energy_fraction(data.telemetry),
        boost_active=data.boost_active,
        boost_lamp_level=data.boost_lamp_level,
        policy_deterministic=data.policy_deterministic,
        policy_action=data.policy_action,
        action_mask_branches=data.action_mask_branches,
        continuous_drive_deadzone=data.continuous_drive_deadzone,
        continuous_drive_enabled=data.continuous_drive_enabled,
        force_full_throttle=data.force_full_throttle,
        continuous_pitch_enabled=data.continuous_pitch_enabled,
        continuous_air_brake_axis_index=data.continuous_air_brake_axis_index,
        continuous_air_brake_deadzone=data.continuous_air_brake_deadzone,
        continuous_air_brake_full_threshold=data.continuous_air_brake_full_threshold,
        continuous_air_brake_min_duty=data.continuous_air_brake_min_duty,
        continuous_air_brake_mode=data.continuous_air_brake_mode,
        continuous_air_brake_disabled=data.continuous_air_brake_disabled,
        spin_requested=_bool_info(data.info, "spin_requested"),
        spin_request=_spin_request_info(data.info),
        spin_macro_active=_bool_info(data.info, "spin_macro_active"),
        spin_macro_cooldown_frames=_int_info(data.info, "spin_macro_cooldown_frames"),
    )
    control_hitboxes, control_bottom = _draw_control_viz_below_game(
        pygame=pygame,
        screen=screen,
        fonts=fonts,
        left_column_size=(left_column_width, game_display_size[1]),
        control_viz=control_viz,
    )
    observation_surface = None
    if data.policy_observation_image is not None:
        preview_frame = _preview_frame(data.policy_observation_image, info=data.info)
        observation_display_size = _observation_preview_size(
            observation_layout_shape,
            info=data.info,
        )
        observation_surface = _rgb_surface(pygame, preview_frame)
        if observation_surface.get_size() != observation_display_size:
            raise RuntimeError(
                "Native observation preview size did not match the expected preview size: "
                f"frame={observation_surface.get_size()}, expected={observation_display_size}"
            )
    _draw_observation_preview_in_rect(
        pygame=pygame,
        screen=screen,
        fonts=fonts,
        surface=observation_surface,
        x=LAYOUT.preview_padding,
        y=control_bottom + LAYOUT.preview_gap,
        width=left_column_width - (2 * LAYOUT.preview_padding),
        height=screen.get_height() - control_bottom - LAYOUT.preview_gap - LAYOUT.preview_padding,
        observation_shape=data.policy_observation_shape,
        layout_shape=observation_layout_shape,
        info=data.info,
    )
    panel_rect = pygame.Rect(
        left_column_width + LAYOUT.preview_gap,
        0,
        LAYOUT.panel_width,
        _watch_window_size(
            game_display_size,
            observation_layout_shape,
            fonts=fonts,
            info=data.info,
            panel_tab_index=data.panel_tab_index,
        )[1],
    )
    side_panel_hitboxes = _draw_side_panel(
        pygame=pygame,
        screen=screen,
        fonts=fonts,
        panel_rect=panel_rect,
        data=SidePanelData(
            episode=data.episode,
            info=data.info,
            reset_info=data.reset_info,
            episode_reward=data.episode_reward,
            paused=data.paused,
            control_state=data.control_state,
            gas_level=data.gas_level,
            thrust_warning_threshold=data.thrust_warning_threshold,
            boost_active=data.boost_active,
            boost_lamp_level=data.boost_lamp_level,
            action_mask_branches=data.action_mask_branches,
            policy_label=data.policy_label,
            policy_curriculum_stage=data.policy_curriculum_stage,
            policy_num_timesteps=data.policy_num_timesteps,
            policy_experience_frames=data.policy_experience_frames,
            policy_deterministic=data.policy_deterministic,
            manual_control_enabled=data.manual_control_enabled,
            policy_action=data.policy_action,
            policy_reload_age_seconds=data.policy_reload_age_seconds,
            policy_reload_error=data.policy_reload_error,
            cnn_activations=data.cnn_activations,
            best_finish_position=data.best_finish_position,
            best_finish_ranks=data.best_finish_ranks,
            best_finish_rank_times=data.best_finish_rank_times,
            best_finish_rank_setups=data.best_finish_rank_setups,
            best_finish_times=data.best_finish_times,
            best_finish_time_ranks=data.best_finish_time_ranks,
            best_finish_time_setups=data.best_finish_time_setups,
            latest_finish_times=data.latest_finish_times,
            latest_finish_deltas_ms=data.latest_finish_deltas_ms,
            track_attempt_stats=data.track_attempt_stats,
            failed_track_attempts=data.failed_track_attempts,
            track_pool_records=data.track_pool_records,
            panel_tab_index=data.panel_tab_index,
            cnn_layer_tab_index=data.cnn_layer_tab_index,
            record_tab_index=data.record_tab_index,
            panel_tabs=data.panel_tabs,
            continuous_drive_deadzone=data.continuous_drive_deadzone,
            continuous_air_brake_axis_index=data.continuous_air_brake_axis_index,
            continuous_air_brake_deadzone=data.continuous_air_brake_deadzone,
            continuous_air_brake_full_threshold=data.continuous_air_brake_full_threshold,
            continuous_air_brake_min_duty=data.continuous_air_brake_min_duty,
            continuous_air_brake_mode=data.continuous_air_brake_mode,
            continuous_air_brake_disabled=data.continuous_air_brake_disabled,
            action_repeat=data.action_repeat,
            max_episode_steps=data.max_episode_steps,
            progress_frontier_stall_limit_frames=data.progress_frontier_stall_limit_frames,
            stuck_min_speed_kph=data.stuck_min_speed_kph,
            game_display_size=game_display_size,
            observation_shape=data.policy_observation_shape,
            observation_state=data.observation_state,
            observation_state_reference=data.observation_state_reference,
            observation_state_feature_names=data.observation_state_feature_names,
            policy_auxiliary_state_predictions=data.policy_auxiliary_state_predictions,
            policy_auxiliary_state_targets=data.policy_auxiliary_state_targets,
            auxiliary_episode_metrics=data.auxiliary_episode_metrics,
            live_episode_series=data.live_episode_series,
            telemetry=data.telemetry,
            emulator_renderer=data.emulator_renderer,
            watch_device=data.watch_device,
            train_config=data.train_config,
            policy_config=data.policy_config,
        ),
    )
    pygame.display.flip()
    return ViewerHitboxes(
        deterministic_toggle=control_hitboxes.deterministic_toggle,
        panel_tabs=side_panel_hitboxes.panel_tabs,
        cnn_layer_tabs=side_panel_hitboxes.cnn_layer_tabs,
        record_tabs=side_panel_hitboxes.record_tabs,
        record_courses=side_panel_hitboxes.record_courses,
        state_features=side_panel_hitboxes.state_features,
    )


def _rgb_surface(pygame: PygameModule, frame: RgbFrame) -> PygameSurface:
    rgb_frame = np.ascontiguousarray(frame)
    height, width, channels = rgb_frame.shape
    if channels != 3:
        raise ValueError(f"Expected an RGB frame for display, got shape {frame.shape!r}")
    return pygame.image.frombuffer(rgb_frame.tobytes(), (width, height), "RGB")


def _engine_setting_level(info: dict[str, object]) -> float | None:
    value = info.get("engine_setting_ram")
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    level = float(value)
    if not math.isfinite(level):
        return None
    return max(0.0, min(1.0, level))


def _speed_kph(telemetry: FZeroXTelemetry | None) -> float | None:
    if telemetry is None:
        return None
    speed = float(telemetry.player.speed_kph)
    if not math.isfinite(speed):
        return None
    return max(0.0, speed)


def _energy_fraction(telemetry: FZeroXTelemetry | None) -> float | None:
    if telemetry is None:
        return None
    max_energy = float(telemetry.player.max_energy)
    if max_energy <= 0.0:
        return None
    energy = float(telemetry.player.energy)
    if not math.isfinite(energy) or not math.isfinite(max_energy):
        return None
    return max(0.0, min(1.0, energy / max_energy))


def _game_course_overlay_label(
    info: dict[str, object],
    *,
    reset_info: dict[str, object] | None = None,
) -> str | None:
    course_name = info.get("track_course_name")
    if not isinstance(course_name, str) or not course_name:
        course_name = _fallback_course_name(info)
    if course_name is None:
        return None

    difficulty = _course_difficulty_name(info)
    course_label = course_name if difficulty is None else f"{course_name} · {difficulty}"
    cup = _course_cup_name(info)
    label = course_label if cup is None else f"{cup} : {course_label}"
    if _course_anchor_active(info, reset_info=reset_info):
        return f"> {label} <"
    return label


def _course_anchor_active(
    info: dict[str, object],
    *,
    reset_info: dict[str, object] | None,
) -> bool:
    if reset_info is None:
        return False
    locked_target_key = reset_info.get("track_sampling_locked_reset_target_key")
    if not isinstance(locked_target_key, str) or not locked_target_key:
        locked_target_key = reset_info.get("track_sampling_locked_course_id")
    if not isinstance(locked_target_key, str) or not locked_target_key:
        return False
    target_key = info.get("track_reset_target_key")
    if not isinstance(target_key, str) or not target_key:
        target_key = info.get("track_reset_course_key")
    if not isinstance(target_key, str) or not target_key:
        target_key = info.get("track_course_key")
    if not isinstance(target_key, str) or not target_key:
        target_key = info.get("track_runtime_course_key")
    if not isinstance(target_key, str) or not target_key:
        target_key = info.get("track_course_id")
    if target_key is None:
        return True
    return target_key == locked_target_key


def _game_speed_overlay_label(info: dict[str, object], *, action_repeat: int) -> str | None:
    native_fps = _finite_float_info(info, "native_fps")
    if native_fps is None or native_fps <= 0.0:
        return None

    actual_game_fps = _finite_float_info(info, "game_fps")
    if actual_game_fps is None or actual_game_fps <= 0.0:
        control_fps = _finite_float_info(info, "control_fps")
        if control_fps is None or control_fps <= 0.0:
            return None
        actual_game_fps = control_fps * max(1, int(action_repeat))

    speedup = min(99.9, max(0.0, actual_game_fps / native_fps))
    return f"{speedup:.1f}x"


def _finite_float_info(info: dict[str, object], key: str) -> float | None:
    value = info.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _bool_info(info: dict[str, object], key: str) -> bool:
    return info.get(key) is True


def _int_info(info: dict[str, object], key: str) -> int:
    value = info.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return max(0, value)


def _spin_request_info(info: dict[str, object]) -> str:
    value = info.get("spin_request")
    return value if isinstance(value, str) and value in {"left", "right"} else "none"


def _course_cup_name(info: dict[str, object]) -> str | None:
    course_ref = info.get("track_course_ref")
    if isinstance(course_ref, str) and "/" in course_ref:
        cup = course_ref.split("/", maxsplit=1)[0].strip()
        if cup:
            cup_label = _format_track_label(cup)
            return cup_label if cup_label.lower().endswith("cup") else f"{cup_label} Cup"

    course_index = info.get("track_course_index", info.get("course_index"))
    if isinstance(course_index, bool) or not isinstance(course_index, int):
        return None
    cups = ("Jack", "Queen", "King", "Joker")
    cup_index = course_index // 6
    if 0 <= cup_index < len(cups):
        return f"{cups[cup_index]} Cup"
    return None


def _course_difficulty_name(info: dict[str, object]) -> str | None:
    for key in (
        "track_gp_difficulty",
        "track_source_gp_difficulty",
        "watch_selected_gp_difficulty",
        "gp_difficulty",
        "source_gp_difficulty",
    ):
        value = info.get(key)
        if isinstance(value, str) and value:
            return _format_track_label(value)
    return None


def _fallback_course_name(info: dict[str, object]) -> str | None:
    display_name = info.get("track_display_name")
    if isinstance(display_name, str) and display_name:
        return _short_track_name(display_name)

    course_id = info.get("track_course_id")
    if isinstance(course_id, str) and course_id:
        return _format_track_label(course_id)

    course_index = info.get("track_course_index", info.get("course_index"))
    if isinstance(course_index, bool):
        return None
    if isinstance(course_index, int):
        return f"course {course_index}"
    return None


def _short_track_name(value: str) -> str:
    suffixes = (
        " Time Attack - Blue Falcon Engine 50",
        " GP Race - Blue Falcon Engine 50",
        " time attack - blue falcon engine 50",
        " gp race - blue falcon engine 50",
    )
    for suffix in suffixes:
        if value.endswith(suffix):
            return value[: -len(suffix)]
    return value


def _format_track_label(value: str) -> str:
    return value.replace("_", " ").title()


def _apply_window_position_hint() -> None:
    os.environ["SDL_VIDEO_WINDOW_POS"] = "100,100"
