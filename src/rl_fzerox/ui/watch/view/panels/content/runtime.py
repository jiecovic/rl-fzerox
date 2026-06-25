# src/rl_fzerox/ui/watch/view/panels/content/runtime.py
"""Run, episode, and runtime side-panel sections for the Watch UI."""

from __future__ import annotations

from rl_fzerox.ui.watch.view.panels.core.format import (
    _float_info,
    _format_checkpoint_experience,
    _format_env_step,
    _format_episode_frames,
    _format_game_speed,
    _format_height_width,
    _format_observation_shape,
    _format_progress_frontier_counter,
    _format_reload_age,
    _int_info,
)
from rl_fzerox.ui.watch.view.panels.core.lines import panel_line as _panel_line
from rl_fzerox.ui.watch.view.screen.theme import PALETTE
from rl_fzerox.ui.watch.view.screen.types import PanelSection


def run_state_section(
    *,
    paused: bool,
    policy_label: str | None,
    manual_control_enabled: bool,
    policy_experience_frames: int | None,
    policy_reload_age_seconds: float | None,
) -> PanelSection:
    return PanelSection(
        title="Run State",
        lines=[
            _panel_line(
                "State",
                "paused" if paused else "running",
                PALETTE.text_warning if paused else PALETTE.text_accent,
            ),
            _panel_line(
                "Driver",
                _format_driver_mode(
                    policy_label=policy_label,
                    manual_control_enabled=manual_control_enabled,
                ),
                _driver_mode_color(
                    policy_label=policy_label,
                    manual_control_enabled=manual_control_enabled,
                ),
            ),
            _panel_line(
                "Experience",
                _format_checkpoint_experience(policy_experience_frames),
                PALETTE.text_primary
                if policy_experience_frames is not None
                else PALETTE.text_muted,
            ),
            _panel_line(
                "Reload",
                _format_reload_age(policy_reload_age_seconds),
                PALETTE.text_primary,
            ),
        ],
    )


def episode_progress_section(
    *,
    episode: int,
    info: dict[str, object],
    episode_reward: float,
    action_repeat: int,
    max_episode_steps: int,
    progress_frontier_stall_limit_frames: int | None,
) -> PanelSection:
    return PanelSection(
        title="Episode Progress",
        lines=[
            _panel_line("Episode", str(episode), PALETTE.text_primary),
            _panel_line(
                "Episode frame",
                _format_episode_frames(info, max_episode_steps=max_episode_steps),
                PALETTE.text_primary,
            ),
            _panel_line(
                "Env step",
                _format_env_step(
                    info,
                    action_repeat=action_repeat,
                    max_episode_steps=max_episode_steps,
                ),
                PALETTE.text_primary,
            ),
            _panel_line(
                "Frontier frames",
                _format_progress_frontier_counter(
                    info,
                    progress_frontier_stall_limit_frames=progress_frontier_stall_limit_frames,
                ),
                PALETTE.text_warning
                if _int_info(info, "progress_frontier_stalled_frames") > 0
                else PALETTE.text_muted,
            ),
            _panel_line(
                "Step reward",
                _format_reward_value(_float_info(info, "step_reward")),
                PALETTE.text_primary,
            ),
            _panel_line(
                "Return",
                _format_reward_value(episode_reward),
                PALETTE.text_primary,
            ),
        ],
    )


def runtime_section(
    *,
    info: dict[str, object],
    emulator_renderer: str,
    action_repeat: int,
    game_display_size: tuple[int, int],
    observation_shape: tuple[int, ...] | None,
) -> PanelSection:
    return PanelSection(
        title="Runtime",
        lines=[
            _panel_line("Renderer", emulator_renderer, PALETTE.text_primary),
            _panel_line(
                "Repeat",
                f"x{action_repeat}",
                PALETTE.text_primary,
            ),
            _panel_line(
                "Control FPS",
                _format_fps_value(info, "control_fps"),
                PALETTE.text_primary,
            ),
            _panel_line(
                "Game FPS",
                _format_fps_value(info, "game_fps"),
                PALETTE.text_primary,
            ),
            _panel_line(
                "Render FPS",
                _format_fps_value(info, "render_fps"),
                PALETTE.text_primary,
            ),
            _panel_line(
                "Speed multiplier",
                _format_game_speed(info, action_repeat=action_repeat),
                PALETTE.text_primary,
            ),
            _panel_line(
                "Game size",
                _format_height_width(game_display_size[1], game_display_size[0]),
                PALETTE.text_primary,
            ),
            _panel_line(
                "Obs size",
                "-" if observation_shape is None else _format_observation_shape(observation_shape),
                PALETTE.text_muted if observation_shape is None else PALETTE.text_primary,
            ),
        ],
    )


def _format_reward_value(value: float) -> str:
    return f"{value:.4f}"


def _format_fps_value(info: dict[str, object], key: str) -> str:
    return f"{_float_info(info, key):.1f}"


def _format_driver_mode(*, policy_label: str | None, manual_control_enabled: bool) -> str:
    if policy_label is None or manual_control_enabled:
        return "manual"
    return "policy"


def _driver_mode_color(
    *,
    policy_label: str | None,
    manual_control_enabled: bool,
) -> tuple[int, int, int]:
    if manual_control_enabled:
        return PALETTE.text_warning
    if policy_label is not None:
        return PALETTE.text_accent
    return PALETTE.text_primary
