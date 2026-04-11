# src/rl_fzerox/ui/watch/render/frame.py
from __future__ import annotations

import os

import numpy as np

from fzerox_emulator import display_size
from rl_fzerox.ui.watch.hud.draw import _draw_side_panel
from rl_fzerox.ui.watch.hud.format import _display_aspect_ratio
from rl_fzerox.ui.watch.hud.model import (
    _observation_preview_size,
    _preview_frame,
    _window_size,
)
from rl_fzerox.ui.watch.layout import FONT_SIZES, LAYOUT, PALETTE, ViewerFonts


def _create_fonts(pygame) -> ViewerFonts:
    return ViewerFonts(
        title=pygame.font.Font(None, FONT_SIZES.title),
        section=pygame.font.Font(None, FONT_SIZES.section),
        body=pygame.font.Font(None, FONT_SIZES.body),
        small=pygame.font.Font(None, FONT_SIZES.small),
    )


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
    if screen.get_size() == _window_size(game_display_size, observation_shape):
        return screen
    return _create_screen(pygame, game_display_size, observation_shape)


def _draw_frame(
    *,
    pygame,
    screen,
    fonts,
    raw_frame: np.ndarray,
    observation: np.ndarray,
    observation_state: np.ndarray | None,
    observation_state_feature_names: tuple[str, ...],
    episode: int,
    info: dict[str, object],
    reset_info: dict[str, object],
    episode_reward: float,
    paused: bool,
    control_state,
    policy_label: str | None,
    policy_curriculum_stage: str | None,
    policy_deterministic: bool | None,
    policy_action: np.ndarray | None,
    policy_reload_age_seconds: float | None,
    policy_reload_error: str | None,
    action_repeat: int,
    max_episode_steps: int,
    stuck_step_limit: int,
    wrong_way_timer_limit: int,
    progress_frontier_stall_limit_frames: int | None,
    stuck_min_speed_kph: float,
    telemetry,
) -> None:
    game_surface = _rgb_surface(pygame, raw_frame)
    game_display_size = display_size(raw_frame.shape, _display_aspect_ratio(info))
    if game_surface.get_size() != game_display_size:
        raise RuntimeError(
            "Native display frame size did not match the expected watch size: "
            f"frame={game_surface.get_size()}, expected={game_display_size}"
        )

    preview_frame = _preview_frame(observation)
    observation_display_size = _observation_preview_size(observation.shape)
    observation_surface = _rgb_surface(pygame, preview_frame)
    if observation_surface.get_size() != observation_display_size:
        raise RuntimeError(
            "Native observation preview size did not match the expected preview size: "
            f"frame={observation_surface.get_size()}, expected={observation_display_size}"
        )

    screen.fill(PALETTE.app_background)
    screen.blit(game_surface, (0, 0))
    pygame.draw.rect(
        screen,
        PALETTE.text_warning,
        pygame.Rect(0, 0, game_display_size[0], game_display_size[1]),
        width=2,
        border_radius=4,
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
        episode=episode,
        info=info,
        reset_info=reset_info,
        episode_reward=episode_reward,
        paused=paused,
        control_state=control_state,
        policy_label=policy_label,
        policy_curriculum_stage=policy_curriculum_stage,
        policy_deterministic=policy_deterministic,
        policy_action=policy_action,
        policy_reload_age_seconds=policy_reload_age_seconds,
        policy_reload_error=policy_reload_error,
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
        observation_surface=observation_surface,
        telemetry=telemetry,
    )
    pygame.display.flip()


def _rgb_surface(pygame, frame: np.ndarray):
    rgb_frame = np.ascontiguousarray(frame)
    height, width, channels = rgb_frame.shape
    if channels != 3:
        raise ValueError(f"Expected an RGB frame for display, got shape {frame.shape!r}")
    return pygame.image.frombuffer(rgb_frame.tobytes(), (width, height), "RGB")


def _apply_window_position_hint() -> None:
    os.environ["SDL_VIDEO_WINDOW_POS"] = "100,100"
