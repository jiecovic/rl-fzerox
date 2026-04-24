# src/rl_fzerox/ui/watch/view/panels/format.py
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from rl_fzerox.core.domain.hybrid_action import (
    HYBRID_CONTINUOUS_ACTION_KEY,
    HYBRID_DISCRETE_ACTION_KEY,
)
from rl_fzerox.core.envs.actions import ActionValue
from rl_fzerox.ui.watch.view.panels.buttons import BUTTON_LABELS


def _pressed_button_labels(joypad_mask_value: int) -> str:
    pressed = [label for button_id, label in BUTTON_LABELS if joypad_mask_value & (1 << button_id)]
    return " ".join(pressed) if pressed else "none"


def _format_policy_action(policy_action: ActionValue | None) -> str:
    if policy_action is None:
        return "manual"

    if isinstance(policy_action, Mapping):
        continuous = policy_action.get(HYBRID_CONTINUOUS_ACTION_KEY)
        discrete = policy_action.get(HYBRID_DISCRETE_ACTION_KEY)
        if continuous is not None and discrete is not None:
            return f"c={_format_action_values(continuous)} d={_format_action_values(discrete)}"

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


def _format_checkpoint_experience(
    num_timesteps: int | None,
    *,
    action_repeat: int,
) -> str:
    if num_timesteps is None:
        return "-"

    total_frames = max(0, int(num_timesteps)) * max(1, int(action_repeat))
    total_seconds = total_frames // 60
    minutes, _ = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    if days > 0:
        return f"{days}d {hours:02d}h {minutes:02d}m"
    if hours > 0:
        return f"{hours}h {minutes:02d}m"
    return f"{minutes}m"


def _display_aspect_ratio(info: dict[str, object]) -> float:
    value = info.get("display_aspect_ratio")
    if isinstance(value, int | float):
        return float(value)
    return 0.0


def _format_observation_summary(
    observation_shape: tuple[int, ...],
    info: Mapping[str, object] | None = None,
) -> str:
    if len(observation_shape) != 3:
        raise ValueError(f"Expected an HxWxC observation shape, got {observation_shape!r}")
    height, width, channels = observation_shape
    stack_size = _observation_stack_size(observation_shape, info=info)
    stack_mode = _observation_stack_mode(info)
    if stack_mode == "gray":
        color_mode = "gray"
    elif stack_mode == "luma_chroma":
        color_mode = "y+c"
    else:
        color_mode = "rgb"
    if channels == 1:
        color_mode = "gray"
    if _observation_minimap_layer(info):
        color_mode = f"{color_mode}+map"
    return f"{width}x{height} {color_mode} x{stack_size} strip"


def _format_observation_shape(observation_shape: tuple[int, ...]) -> str:
    height, width, channels = observation_shape
    return f"{width}x{height}x{channels}"


def _format_stuck_counter(
    info: dict[str, object],
    *,
    stuck_step_limit: int | None,
) -> str:
    if stuck_step_limit is None:
        return f"{_int_info(info, 'stalled_steps')} / off"
    return f"{_int_info(info, 'stalled_steps')} / {stuck_step_limit}"


def _format_reverse_counter(
    info: dict[str, object],
    *,
    wrong_way_timer_limit: int | None,
) -> str:
    if wrong_way_timer_limit is None:
        return f"{_int_info(info, 'reverse_timer')} / off"
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


def _format_episode_frames(
    info: dict[str, object],
    *,
    max_episode_steps: int,
) -> str:
    return f"{_int_info(info, 'episode_step')} / {max_episode_steps}"


def _format_env_step(
    info: dict[str, object],
    *,
    action_repeat: int,
    max_episode_steps: int,
) -> str:
    repeat = max(1, int(action_repeat))
    episode_frames = _int_info(info, "episode_step")
    env_steps = _ceil_div(episode_frames, repeat)
    max_env_steps = _ceil_div(max_episode_steps, repeat)
    return f"{env_steps} / {max_env_steps}"


def _ceil_div(value: int, divisor: int) -> int:
    if value <= 0:
        return 0
    return (value + divisor - 1) // divisor


def _format_control_rate(info: dict[str, object]) -> str:
    return _format_rate_pair(info, actual_key="control_fps", target_key="control_fps_target")


def _format_game_rate(info: dict[str, object]) -> str:
    return _format_rate_pair(info, actual_key="game_fps", target_key="game_fps_target")


def _format_render_rate(info: dict[str, object]) -> str:
    return _format_rate_pair(info, actual_key="render_fps", target_key="render_fps_target")


def _format_rate_pair(
    info: dict[str, object],
    *,
    actual_key: str,
    target_key: str,
) -> str:
    return f"{_float_info(info, actual_key):.1f} / {_format_rate_target(info, target_key)}"


def _format_rate_target(info: dict[str, object], key: str) -> str:
    value = info.get(key)
    if value == "unlimited":
        return "unlimited"
    if isinstance(value, int | float):
        return f"{float(value):.1f}"
    return "-"


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


def _preview_frame_shape(
    observation_shape: tuple[int, ...],
    info: Mapping[str, object] | None = None,
) -> tuple[int, int, int]:
    if len(observation_shape) != 3:
        raise ValueError(f"Expected an HxWxC observation shape, got {observation_shape!r}")
    height, width, _ = observation_shape
    stack_size = max(1, _observation_stack_size(observation_shape, info=info))
    frame_count = stack_size + (1 if _observation_minimap_layer(info) else 0)
    columns, rows = _observation_preview_grid(frame_count)
    return height * rows, width * columns, 3


def _observation_stack_size(
    observation_shape: tuple[int, ...],
    *,
    info: Mapping[str, object] | None = None,
) -> int:
    if info is not None:
        stack_size = info.get("observation_stack")
        if isinstance(stack_size, int):
            return stack_size
    channels = observation_shape[2]
    if channels % 3 == 0:
        return channels // 3
    return channels


def _observation_stack_mode(info: Mapping[str, object] | None) -> str:
    if info is None:
        return "rgb"
    value = info.get("observation_stack_mode")
    return value if isinstance(value, str) else "rgb"


def _observation_minimap_layer(info: Mapping[str, object] | None) -> bool:
    if info is None:
        return False
    return info.get("observation_minimap_layer") is True


def _observation_preview_grid(stack_size: int) -> tuple[int, int]:
    return stack_size, 1
