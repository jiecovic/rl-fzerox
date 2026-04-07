# src/rl_fzerox/ui/viewer_format.py
from __future__ import annotations

import numpy as np

from rl_fzerox.core.emulator.control import ControllerState
from rl_fzerox.core.game import FZeroXTelemetry
from rl_fzerox.ui.viewer_layout import (
    BUTTON_LABELS,
    FONT_SIZES,
    LAYOUT,
    PALETTE,
    Color,
    PanelColumns,
    PanelLine,
    PanelSection,
    ViewerFonts,
)


def _build_panel_columns(
    *,
    episode: int,
    info: dict[str, object],
    reset_info: dict[str, object],
    episode_reward: float,
    paused: bool,
    control_state: ControllerState,
    policy_label: str | None,
    policy_action: np.ndarray | None,
    policy_reload_age_seconds: float | None,
    policy_reload_error: str | None,
    game_display_size: tuple[int, int],
    observation_shape: tuple[int, ...],
    telemetry: FZeroXTelemetry | None,
) -> PanelColumns:
    return PanelColumns(
        left=[
            PanelSection(
                title="Session",
                lines=[
                    _panel_line(
                        "State",
                        "paused" if paused else "running",
                        PALETTE.text_warning if paused else PALETTE.text_accent,
                    ),
                    _panel_line(
                        "Driver",
                        policy_label if policy_label is not None else "manual",
                        PALETTE.text_primary,
                    ),
                    _panel_line(
                        "Action",
                        _format_policy_action(policy_action),
                        PALETTE.text_primary,
                    ),
                    _panel_line(
                        "Reload",
                        _format_reload_age(policy_reload_age_seconds),
                        PALETTE.text_primary,
                    ),
                    _panel_line(
                        "Reload err",
                        _format_reload_error(policy_reload_error),
                        PALETTE.text_warning
                        if policy_reload_error is not None
                        else PALETTE.text_muted,
                    ),
                    _panel_line("Episode", str(episode), PALETTE.text_primary),
                    _panel_line("Frame", str(info.get("frame_index", 0)), PALETTE.text_primary),
                    _panel_line(
                        "Step",
                        f"{_float_info(info, 'step_reward'):.2f}",
                        PALETTE.text_primary,
                    ),
                    _panel_line("Return", f"{episode_reward:.2f}", PALETTE.text_primary),
                ],
            ),
            PanelSection(
                title="Reset",
                lines=[
                    _panel_line(
                        "Mode",
                        str(reset_info.get("reset_mode", "baseline")),
                        PALETTE.text_primary,
                    ),
                    _panel_line(
                        "Baseline",
                        str(reset_info.get("baseline_kind", "unknown")),
                        PALETTE.text_primary,
                    ),
                    _panel_line(
                        "Boot",
                        str(reset_info.get("boot_state", "-")),
                        PALETTE.text_primary,
                    ),
                ],
            ),
            PanelSection(
                title="Input",
                lines=[
                    _panel_line(
                        "Held",
                        _pressed_button_labels(control_state.joypad_mask),
                        PALETTE.text_primary,
                    ),
                    _panel_line(
                        "Steer",
                        f"{control_state.left_stick_x:+.2f}",
                        PALETTE.text_primary,
                    ),
                ],
            ),
            PanelSection(
                title="Controls",
                lines=[
                    _panel_line("Steer", "Left / Right arrows", PALETTE.text_muted),
                    _panel_line("D-pad", "Arrows", PALETTE.text_muted),
                    _panel_line("A / B", "X / Z", PALETTE.text_muted),
                    _panel_line("Menu", "Enter / Backspace", PALETTE.text_muted),
                    _panel_line("Playback", "P pause | N step", PALETTE.text_muted),
                    _panel_line("Baseline", "K save", PALETTE.text_muted),
                ],
            ),
        ],
        right=[
            _game_section(telemetry),
            PanelSection(
                title="Display",
                lines=[
                    _panel_line(
                        "Game",
                        f"{game_display_size[0]}x{game_display_size[1]}",
                        PALETTE.text_primary,
                    ),
                    _panel_line(
                        "Obs",
                        _format_observation_shape(observation_shape),
                        PALETTE.text_primary,
                    ),
                    _panel_line(
                        "Stack",
                        str(_observation_stack_size(observation_shape)),
                        PALETTE.text_primary,
                    ),
                    _panel_line(
                        "FPS",
                        f"{_float_info(info, 'native_fps'):.1f}",
                        PALETTE.text_primary,
                    ),
                ],
            ),
        ],
    )


def _panel_content_height(fonts: ViewerFonts, columns: PanelColumns) -> int:
    title_surface = fonts.title.render("F-Zero X Watch", True, PALETTE.text_primary)
    subtitle_surface = fonts.small.render("live emulator session", True, PALETTE.text_muted)
    y = LAYOUT.panel_padding
    y += title_surface.get_height() + LAYOUT.title_gap + subtitle_surface.get_height()
    y += LAYOUT.title_section_gap

    left_height = _column_content_height(fonts, columns.left)
    right_height = _column_content_height(fonts, columns.right)
    return y + max(left_height, right_height) + LAYOUT.panel_padding


def _window_size(
    game_display_size: tuple[int, int],
    observation_shape: tuple[int, ...],
) -> tuple[int, int]:
    preview_panel_size = _preview_panel_size(observation_shape)
    return (
        game_display_size[0]
        + LAYOUT.preview_gap
        + preview_panel_size[0]
        + LAYOUT.preview_gap
        + LAYOUT.panel_width,
        max(game_display_size[1], preview_panel_size[1]),
    )


def _preview_frame(observation: np.ndarray) -> np.ndarray:
    if observation.ndim != 3:
        raise ValueError(f"Expected an HxWxC observation, got {observation.shape!r}")

    channels = observation.shape[2]
    if channels == 3:
        return np.ascontiguousarray(observation)
    if channels == 1:
        return np.repeat(observation, 3, axis=2)
    if channels % 3 == 0:
        return np.ascontiguousarray(observation[:, :, -3:])

    latest_channel = observation[:, :, -1:]
    return np.repeat(latest_channel, 3, axis=2)


def _pressed_button_labels(joypad_mask_value: int) -> str:
    pressed = [
        label
        for button_id, label in BUTTON_LABELS
        if joypad_mask_value & (1 << button_id)
    ]
    return " ".join(pressed) if pressed else "none"


def _format_policy_action(policy_action: np.ndarray | None) -> str:
    if policy_action is None:
        return "manual"

    values = np.asarray(policy_action, dtype=np.int64).reshape(-1)
    if len(values) != 2:
        return str(values.tolist())

    steer_bucket, drive_mode = int(values[0]), int(values[1])
    drive_label = "throttle" if drive_mode == 1 else "coast"
    return f"[{steer_bucket},{drive_mode}] {drive_label}"


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
    if len(normalized) <= 28:
        return normalized
    return f"{normalized[:25]}..."


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


def _observation_preview_size(observation_shape: tuple[int, ...]) -> tuple[int, int]:
    preview_shape = _preview_frame_shape(observation_shape)
    return (
        preview_shape[1] * LAYOUT.preview_scale,
        preview_shape[0] * LAYOUT.preview_scale,
    )


def _preview_panel_size(observation_shape: tuple[int, ...]) -> tuple[int, int]:
    preview_width, preview_height = _observation_preview_size(observation_shape)
    title_height = FONT_SIZES.section + LAYOUT.preview_title_gap + FONT_SIZES.small
    panel_height = (
        (2 * LAYOUT.preview_padding)
        + title_height
        + LAYOUT.section_rule_gap
        + preview_height
    )
    panel_width = preview_width + (2 * LAYOUT.preview_padding)
    return panel_width, panel_height


def _column_content_height(fonts: ViewerFonts, sections: list[PanelSection]) -> int:
    y = 0
    for section_index, section in enumerate(sections):
        section_title = fonts.section.render(section.title, True, PALETTE.text_primary)
        y += section_title.get_height() + LAYOUT.section_title_gap
        y += LAYOUT.section_rule_gap
        for line in section.lines:
            if line.label:
                label_surface = fonts.small.render(line.label, True, PALETTE.text_muted)
                value_surface = fonts.body.render(line.value, True, line.color)
                y += max(label_surface.get_height(), value_surface.get_height()) + LAYOUT.line_gap
            else:
                value_surface = fonts.small.render(line.value, True, line.color)
                y += value_surface.get_height() + LAYOUT.line_gap
        if section_index < len(sections) - 1:
            y += LAYOUT.section_gap
    return y


def _panel_line(label: str, value: str, color: Color) -> PanelLine:
    return PanelLine(label=label, value=value, color=color)


def _game_section(telemetry: FZeroXTelemetry | None) -> PanelSection:
    if telemetry is None:
        return PanelSection(
            title="Game",
            lines=[
                _panel_line("Status", "unavailable", PALETTE.text_warning),
            ],
        )

    return PanelSection(
        title="Game",
        lines=[
            _panel_line("Mode", _format_mode_name(telemetry.game_mode_name), PALETTE.text_primary),
            _panel_line("Course", str(telemetry.course_index), PALETTE.text_primary),
            _panel_line(
                "Time",
                _format_race_time_ms(telemetry.player.race_time_ms),
                PALETTE.text_primary,
            ),
            _panel_line("Speed", f"{telemetry.player.speed_kph:.1f} km/h", PALETTE.text_primary),
            _panel_line("Boost", str(telemetry.player.boost_timer), PALETTE.text_primary),
            _panel_line(
                "Energy",
                f"{telemetry.player.energy:.1f} / {telemetry.player.max_energy:.1f}",
                PALETTE.text_primary,
            ),
            _panel_line("Lap", str(telemetry.player.lap), PALETTE.text_primary),
            _panel_line("Pos", str(telemetry.player.position), PALETTE.text_primary),
            _panel_line(
                "Progress",
                _format_distance(telemetry.player.race_distance),
                PALETTE.text_primary,
            ),
            _panel_line(
                "Lap prog",
                _format_distance(telemetry.player.lap_distance),
                PALETTE.text_primary,
            ),
            _panel_line(
                "Lap base",
                _format_distance(telemetry.player.laps_completed_distance),
                PALETTE.text_primary,
            ),
            _panel_line(
                "Sort key",
                _format_distance(telemetry.player.race_distance_position),
                PALETTE.text_primary,
            ),
            _panel_line(
                "Flags",
                _format_state_labels(telemetry.player.state_labels),
                PALETTE.text_primary,
            ),
        ],
    )


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


def _format_observation_shape(observation_shape: tuple[int, ...]) -> str:
    height, width, channels = observation_shape
    return f"{width}x{height}x{channels}"


def _float_info(info: dict[str, object], key: str) -> float:
    value = info.get(key)
    if isinstance(value, int | float):
        return float(value)
    return 0.0


def _format_mode_name(mode_name: str) -> str:
    return mode_name.replace("_", " ")


def _format_race_time_ms(race_time_ms: int) -> str:
    minutes, remainder = divmod(max(0, race_time_ms), 60_000)
    seconds, milliseconds = divmod(remainder, 1_000)
    return f"{minutes:02d}'{seconds:02d}\"{milliseconds:03d}"


def _format_distance(distance: float) -> str:
    return f"{distance:,.1f}"


def _format_state_labels(state_labels: tuple[str, ...]) -> str:
    if not state_labels:
        return "none"
    return " | ".join(state_labels)
