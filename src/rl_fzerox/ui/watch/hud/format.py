# src/rl_fzerox/ui/watch/hud/format.py
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from rl_fzerox.core.envs.actions import ActionValue
from rl_fzerox.ui.watch.layout import BUTTON_LABELS

_RELOAD_ERROR_MAX_CHARS = 36


def _pressed_button_labels(joypad_mask_value: int) -> str:
    pressed = [label for button_id, label in BUTTON_LABELS if joypad_mask_value & (1 << button_id)]
    return " ".join(pressed) if pressed else "none"


def _format_policy_action(policy_action: ActionValue | None) -> str:
    if policy_action is None:
        return "manual"

    if isinstance(policy_action, Mapping):
        continuous = policy_action.get("continuous")
        discrete = policy_action.get("discrete")
        if continuous is not None and discrete is not None:
            return (
                f"c={_format_action_values(continuous)} "
                f"d={_format_action_values(discrete)}"
            )

    return _format_action_values(policy_action)


def _format_action_values(value: object) -> str:
    values = np.asarray(value).reshape(-1)
    if values.dtype == np.dtype("O"):
        return str(value)
    if np.issubdtype(values.dtype, np.floating):
        formatted = [f"{float(value):+.2f}" for value in values]
        return "[" + ",".join(formatted) + "]"
    return str(values.astype(np.int64, copy=False).tolist()).replace(" ", "")


def _format_reload_age(reload_age_seconds: float | None) -> str:
    if reload_age_seconds is None:
        return "manual"

    total_seconds = int(max(0.0, reload_age_seconds))
    if total_seconds < 60:
        return f"{total_seconds}s ago"

    minutes, seconds = divmod(total_seconds, 60)
    if minutes < 60:
        return f"{minutes}m {seconds:02d}s"

    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def _format_reload_error(reload_error: str | None) -> str:
    if reload_error is None:
        return "-"
    normalized = " ".join(reload_error.split())
    if len(normalized) <= _RELOAD_ERROR_MAX_CHARS:
        return normalized
    return normalized[: _RELOAD_ERROR_MAX_CHARS - 1] + "…"


def _display_aspect_ratio(info: dict[str, object]) -> float:
    value = info.get("display_aspect_ratio")
    if isinstance(value, int | float):
        return float(value)
    return 0.0


def _format_observation_summary(observation_shape: tuple[int, ...]) -> str:
    preview_shape = _preview_frame_shape(observation_shape)
    stack_size = _observation_stack_size(observation_shape)
    return (
        f"{preview_shape[1]}x{preview_shape[0]} "
        f"{'rgb' if preview_shape[2] == 3 else 'gray'} "
        f"x{stack_size}"
    )


def _format_observation_shape(observation_shape: tuple[int, ...]) -> str:
    height, width, channels = observation_shape
    return f"{width}x{height}x{channels}"


def _format_stuck_counter(
    info: dict[str, object],
    *,
    stuck_step_limit: int,
) -> str:
    return f"{_int_info(info, 'stalled_steps')} / {stuck_step_limit}"


def _format_reverse_counter(
    info: dict[str, object],
    *,
    wrong_way_timer_limit: int,
) -> str:
    return f"{_int_info(info, 'reverse_timer')} / {wrong_way_timer_limit}"


def _format_progress_frontier_counter(
    info: dict[str, object],
    *,
    progress_frontier_stall_limit_frames: int | None,
) -> str:
    if progress_frontier_stall_limit_frames is None:
        return "-"
    return (
        f"{_int_info(info, 'progress_frontier_stalled_frames')} / "
        f"{progress_frontier_stall_limit_frames}"
    )


def _format_episode_step(
    info: dict[str, object],
    *,
    max_episode_steps: int,
) -> str:
    return f"{_int_info(info, 'episode_step')} / {max_episode_steps}"


def _format_control_game_rate(info: dict[str, object]) -> str:
    return f"{_float_info(info, 'control_fps'):.1f} / {_float_info(info, 'game_fps'):.1f}"


def _float_info(info: dict[str, object], key: str) -> float:
    value = info.get(key)
    if isinstance(value, int | float):
        return float(value)
    return 0.0


def _int_info(info: dict[str, object], key: str) -> int:
    value = info.get(key)
    if isinstance(value, int | float):
        return int(value)
    return 0


def _format_mode_name(mode_name: str) -> str:
    return mode_name.replace("_", " ")


def _format_race_time_ms(race_time_ms: int) -> str:
    minutes, remainder = divmod(max(0, race_time_ms), 60_000)
    seconds, milliseconds = divmod(remainder, 1_000)
    return f"{minutes:02d}'{seconds:02d}\"{milliseconds:03d}"


def _format_distance(distance: float) -> str:
    return f"{distance:,.1f}"


def _preview_frame_shape(observation_shape: tuple[int, ...]) -> tuple[int, int, int]:
    if len(observation_shape) != 3:
        raise ValueError(f"Expected an HxWxC observation shape, got {observation_shape!r}")
    height, width, channels = observation_shape
    preview_channels = 3 if channels % 3 == 0 else 1
    return height, width, preview_channels


def _observation_stack_size(observation_shape: tuple[int, ...]) -> int:
    channels = observation_shape[2]
    if channels % 3 == 0:
        return channels // 3
    return channels
