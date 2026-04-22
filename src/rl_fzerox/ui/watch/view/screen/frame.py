# src/rl_fzerox/ui/watch/view/screen/frame.py
from __future__ import annotations

import math
import os

import numpy as np

from fzerox_emulator import FZeroXTelemetry
from fzerox_emulator.arrays import ObservationFrame, RgbFrame, StateVector
from rl_fzerox.core.envs.actions import ActionValue
from rl_fzerox.core.envs.engine.controls import ActionMaskBranches
from rl_fzerox.ui.watch.view.components.game_view import _draw_glass_game_view
from rl_fzerox.ui.watch.view.components.observation_strip import (
    _draw_observation_preview_below_game,
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
from rl_fzerox.ui.watch.view.screen.types import ViewerFonts, ViewerHitboxes


def _create_fonts(pygame) -> ViewerFonts:
    record_header = pygame.font.Font(None, FONT_SIZES.small)
    record_header.set_bold(True)
    return ViewerFonts(
        title=pygame.font.Font(None, FONT_SIZES.title),
        section=pygame.font.Font(None, FONT_SIZES.section),
        record_header=record_header,
        body=_create_mono_font(pygame, FONT_SIZES.body),
        small=pygame.font.Font(None, FONT_SIZES.small),
    )


def _create_mono_font(pygame, size: int):
    font_path = pygame.font.match_font(
        ("dejavusansmono", "liberationmono", "consolas", "couriernew", "monospace")
    )
    if font_path is not None:
        return pygame.font.Font(font_path, size)
    return pygame.font.SysFont("monospace", size)


def _create_screen(
    pygame,
    game_display_size: tuple[int, int],
    observation_shape: tuple[int, ...],
):
    _apply_window_position_hint()
    screen = pygame.display.set_mode(_window_size(game_display_size, observation_shape))
    pygame.display.set_caption("F-Zero X Watch")
    return screen


def _ensure_screen(
    pygame,
    screen,
    game_display_size: tuple[int, int],
    observation_shape: tuple[int, ...],
):
    if screen is None:
        return _create_screen(pygame, game_display_size, observation_shape)
    if screen.get_size() == _window_size(game_display_size, observation_shape):
        return screen
    return _create_screen(pygame, game_display_size, observation_shape)


def _watch_game_display_size() -> tuple[int, int]:
    return LAYOUT.game_display_size


def _draw_frame(
    *,
    pygame,
    screen,
    fonts,
    raw_frame: RgbFrame,
    observation: ObservationFrame,
    observation_state: StateVector | None,
    observation_state_feature_names: tuple[str, ...],
    episode: int,
    info: dict[str, object],
    reset_info: dict[str, object],
    episode_reward: float,
    paused: bool,
    control_state,
    gas_level: float,
    thrust_warning_threshold: float | None,
    thrust_deadzone_threshold: float | None,
    thrust_full_threshold: float | None,
    boost_active: bool,
    boost_lamp_level: float,
    action_mask_branches: ActionMaskBranches,
    policy_label: str | None,
    policy_curriculum_stage: str | None,
    policy_deterministic: bool | None,
    policy_action: ActionValue | None,
    policy_reload_age_seconds: float | None,
    policy_reload_error: str | None,
    best_finish_position: int | None,
    best_finish_times: dict[str, int],
    latest_finish_times: dict[str, int],
    latest_finish_deltas_ms: dict[str, int],
    track_pool_records: tuple[dict[str, object], ...],
    continuous_drive_deadzone: float,
    continuous_air_brake_mode: str,
    continuous_air_brake_disabled: bool,
    action_repeat: int,
    max_episode_steps: int,
    stuck_step_limit: int | None,
    wrong_way_timer_limit: int | None,
    progress_frontier_stall_limit_frames: int | None,
    stuck_min_speed_kph: float,
    telemetry: FZeroXTelemetry | None,
) -> ViewerHitboxes:
    game_display_size = _watch_game_display_size()
    game_surface = _rgb_surface(pygame, raw_frame)

    preview_frame = _preview_frame(observation, info=info)
    observation_display_size = _observation_preview_size(observation.shape, info=info)
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
    hitboxes = _draw_observation_preview_below_game(
        pygame=pygame,
        screen=screen,
        fonts=fonts,
        surface=observation_surface,
        game_display_size=game_display_size,
        observation_shape=observation.shape,
        info=info,
        control_viz=_control_viz(
            control_state,
            gas_level=gas_level,
            thrust_warning_threshold=thrust_warning_threshold,
            thrust_deadzone_threshold=thrust_deadzone_threshold,
            thrust_full_threshold=thrust_full_threshold,
            engine_setting_level=_engine_setting_level(info),
            speed_kph=_speed_kph(telemetry),
            energy_fraction=_energy_fraction(telemetry),
            boost_active=boost_active,
            boost_lamp_level=boost_lamp_level,
            policy_deterministic=policy_deterministic,
            policy_action=policy_action,
            action_mask_branches=action_mask_branches,
            continuous_drive_deadzone=continuous_drive_deadzone,
            continuous_air_brake_mode=continuous_air_brake_mode,
            continuous_air_brake_disabled=continuous_air_brake_disabled,
        ),
    )
    panel_rect = pygame.Rect(
        game_display_size[0] + LAYOUT.preview_gap,
        0,
        LAYOUT.panel_width,
        _window_size(game_display_size, observation.shape)[1],
    )
    _draw_side_panel(
        pygame=pygame,
        screen=screen,
        fonts=fonts,
        panel_rect=panel_rect,
        data=SidePanelData(
            episode=episode,
            info=info,
            reset_info=reset_info,
            episode_reward=episode_reward,
            paused=paused,
            control_state=control_state,
            gas_level=gas_level,
            thrust_warning_threshold=thrust_warning_threshold,
            boost_active=boost_active,
            boost_lamp_level=boost_lamp_level,
            action_mask_branches=action_mask_branches,
            policy_label=policy_label,
            policy_curriculum_stage=policy_curriculum_stage,
            policy_deterministic=policy_deterministic,
            policy_action=policy_action,
            policy_reload_age_seconds=policy_reload_age_seconds,
            policy_reload_error=policy_reload_error,
            best_finish_position=best_finish_position,
            best_finish_times=best_finish_times,
            latest_finish_times=latest_finish_times,
            latest_finish_deltas_ms=latest_finish_deltas_ms,
            track_pool_records=track_pool_records,
            continuous_drive_deadzone=continuous_drive_deadzone,
            continuous_air_brake_mode=continuous_air_brake_mode,
            continuous_air_brake_disabled=continuous_air_brake_disabled,
            action_repeat=action_repeat,
            max_episode_steps=max_episode_steps,
            stuck_step_limit=stuck_step_limit,
            wrong_way_timer_limit=wrong_way_timer_limit,
            progress_frontier_stall_limit_frames=progress_frontier_stall_limit_frames,
            stuck_min_speed_kph=stuck_min_speed_kph,
            game_display_size=game_display_size,
            observation_shape=observation.shape,
            observation_state=observation_state,
            observation_state_feature_names=observation_state_feature_names,
            telemetry=telemetry,
        ),
    )
    pygame.display.flip()
    return hitboxes


def _rgb_surface(pygame, frame: RgbFrame):
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
