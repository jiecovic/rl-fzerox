# src/rl_fzerox/ui/watch/view/screen/frame.py
from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np

from fzerox_emulator import ControllerState, FZeroXTelemetry
from fzerox_emulator.arrays import ObservationFrame, RgbFrame, StateVector
from rl_fzerox.core.config.schema import PolicyConfig, TrainConfig
from rl_fzerox.core.envs.actions import ActionValue
from rl_fzerox.core.envs.engine.controls import ActionMaskBranches
from rl_fzerox.ui.watch.runtime.cnn import CnnActivationSnapshot
from rl_fzerox.ui.watch.view.components.game_view import _draw_glass_game_view
from rl_fzerox.ui.watch.view.components.observation_strip import (
    _draw_control_viz_below_game,
    _draw_observation_preview_in_rect,
)
from rl_fzerox.ui.watch.view.panels.draw import SidePanelData, _draw_side_panel
from rl_fzerox.ui.watch.view.panels.model import (
    _observation_preview_size,
    _preview_frame,
    _window_size,
)
from rl_fzerox.ui.watch.view.panels.viz import _control_viz
from rl_fzerox.ui.watch.view.screen.layout import LAYOUT
from rl_fzerox.ui.watch.view.screen.theme import FONT_SIZES, PALETTE
from rl_fzerox.ui.watch.view.screen.types import (
    PygameModule,
    PygameSurface,
    RenderFont,
    ViewerFonts,
    ViewerHitboxes,
)


@dataclass(frozen=True, slots=True)
class FrameRenderData:
    """Complete state snapshot needed to render one watch frame."""

    raw_frame: RgbFrame
    observation: ObservationFrame
    observation_state: StateVector | None
    observation_state_feature_names: tuple[str, ...]
    episode: int
    info: dict[str, object]
    reset_info: dict[str, object]
    episode_reward: float
    paused: bool
    control_state: ControllerState
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
    policy_deterministic: bool | None
    policy_action: ActionValue | None
    policy_reload_age_seconds: float | None
    policy_reload_error: str | None
    cnn_activations: CnnActivationSnapshot | None
    best_finish_position: int | None
    best_finish_times: dict[str, int]
    latest_finish_times: dict[str, int]
    latest_finish_deltas_ms: dict[str, int]
    track_pool_records: tuple[dict[str, object], ...]
    panel_tab_index: int
    continuous_drive_deadzone: float
    continuous_air_brake_mode: str
    continuous_air_brake_disabled: bool
    action_repeat: int
    max_episode_steps: int
    progress_frontier_stall_limit_frames: int | None
    stuck_min_speed_kph: float
    telemetry: FZeroXTelemetry | None
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
    panel_tab_index: int = 0,
) -> PygameSurface:
    _apply_window_position_hint()
    screen = pygame.display.set_mode(
        _window_size(
            game_display_size,
            observation_shape,
            panel_tab_index=panel_tab_index,
        )
    )
    pygame.display.set_caption("F-Zero X Watch")
    return screen


def _ensure_screen(
    pygame: PygameModule,
    screen: PygameSurface | None,
    game_display_size: tuple[int, int],
    observation_shape: tuple[int, ...],
    *,
    panel_tab_index: int = 0,
) -> PygameSurface:
    if screen is None:
        return _create_screen(
            pygame,
            game_display_size,
            observation_shape,
            panel_tab_index=panel_tab_index,
        )
    if screen.get_size() == _window_size(
        game_display_size,
        observation_shape,
        panel_tab_index=panel_tab_index,
    ):
        return screen
    return _create_screen(
        pygame,
        game_display_size,
        observation_shape,
        panel_tab_index=panel_tab_index,
    )


def _watch_game_display_size() -> tuple[int, int]:
    return LAYOUT.game_display_size


def _draw_frame(
    *,
    pygame: PygameModule,
    screen: PygameSurface,
    fonts: ViewerFonts,
    data: FrameRenderData,
) -> ViewerHitboxes:
    game_display_size = _watch_game_display_size()
    game_surface = _rgb_surface(pygame, data.raw_frame)

    preview_frame = _preview_frame(data.observation, info=data.info)
    observation_display_size = _observation_preview_size(data.observation.shape, info=data.info)
    observation_surface = _rgb_surface(pygame, preview_frame)
    if observation_surface.get_size() != observation_display_size:
        raise RuntimeError(
            "Native observation preview size did not match the expected preview size: "
            f"frame={observation_surface.get_size()}, expected={observation_display_size}"
        )

    screen.fill(PALETTE.app_background)
    _draw_glass_game_view(
        pygame=pygame,
        screen=screen,
        surface=game_surface,
        outer_size=game_display_size,
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
        continuous_air_brake_mode=data.continuous_air_brake_mode,
        continuous_air_brake_disabled=data.continuous_air_brake_disabled,
    )
    control_hitboxes, control_bottom = _draw_control_viz_below_game(
        pygame=pygame,
        screen=screen,
        fonts=fonts,
        game_display_size=game_display_size,
        control_viz=control_viz,
    )
    _draw_observation_preview_in_rect(
        pygame=pygame,
        screen=screen,
        fonts=fonts,
        surface=observation_surface,
        x=LAYOUT.preview_padding,
        y=control_bottom + LAYOUT.preview_gap,
        width=game_display_size[0] - (2 * LAYOUT.preview_padding),
        height=screen.get_height() - control_bottom - LAYOUT.preview_gap - LAYOUT.preview_padding,
        observation_shape=data.observation.shape,
        info=data.info,
    )
    panel_rect = pygame.Rect(
        game_display_size[0] + LAYOUT.preview_gap,
        0,
        LAYOUT.panel_width,
        _window_size(
            game_display_size,
            data.observation.shape,
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
            policy_deterministic=data.policy_deterministic,
            policy_action=data.policy_action,
            policy_reload_age_seconds=data.policy_reload_age_seconds,
            policy_reload_error=data.policy_reload_error,
            cnn_activations=data.cnn_activations,
            best_finish_position=data.best_finish_position,
            best_finish_times=data.best_finish_times,
            latest_finish_times=data.latest_finish_times,
            latest_finish_deltas_ms=data.latest_finish_deltas_ms,
            track_pool_records=data.track_pool_records,
            panel_tab_index=data.panel_tab_index,
            continuous_drive_deadzone=data.continuous_drive_deadzone,
            continuous_air_brake_mode=data.continuous_air_brake_mode,
            continuous_air_brake_disabled=data.continuous_air_brake_disabled,
            action_repeat=data.action_repeat,
            max_episode_steps=data.max_episode_steps,
            progress_frontier_stall_limit_frames=data.progress_frontier_stall_limit_frames,
            stuck_min_speed_kph=data.stuck_min_speed_kph,
            game_display_size=game_display_size,
            observation_shape=data.observation.shape,
            observation_state=data.observation_state,
            observation_state_feature_names=data.observation_state_feature_names,
            telemetry=data.telemetry,
            train_config=data.train_config,
            policy_config=data.policy_config,
        ),
    )
    pygame.display.flip()
    return ViewerHitboxes(
        deterministic_toggle=control_hitboxes.deterministic_toggle,
        panel_tabs=side_panel_hitboxes.panel_tabs,
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


def _apply_window_position_hint() -> None:
    os.environ["SDL_VIDEO_WINDOW_POS"] = "100,100"
