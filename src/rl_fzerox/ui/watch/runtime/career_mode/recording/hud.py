# src/rl_fzerox/ui/watch/runtime/career_mode/recording/hud.py
from __future__ import annotations

from collections.abc import Mapping
from typing import TypeAlias

import numpy as np

from fzerox_emulator.arrays import RgbFrame

_RgbColor: TypeAlias = tuple[int, int, int]

INPUT_HUD_INFO_KEYS = (
    "watch_recording_input_hud",
    "watch_recording_input_gas",
    "watch_recording_input_boost",
    "watch_recording_input_air_brake",
    "watch_recording_input_lean_left",
    "watch_recording_input_lean_right",
    "watch_recording_input_stick_x",
    "watch_recording_input_pitch",
)


_HUD_TINY_GLYPHS: dict[str, tuple[str, ...]] = {
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    "B": ("11110", "10001", "10001", "11110", "10001", "10001", "11110"),
    "C": ("01111", "10000", "10000", "10000", "10000", "10000", "01111"),
    "D": ("11110", "10001", "10001", "10001", "10001", "10001", "11110"),
    "L": ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
    "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    "Z": ("11111", "00001", "00010", "00100", "01000", "10000", "11111"),
}


def input_hud_frame(frame: RgbFrame, info: Mapping[str, object]) -> RgbFrame:
    """Return a frame with the optional recording input HUD rendered on top."""

    if info.get("watch_recording_input_hud") is not True:
        return frame
    height, width = frame.shape[:2]
    if width < 120 or height < 90:
        return frame
    hud_frame: RgbFrame = np.array(frame, copy=True)
    _draw_input_hud(hud_frame, info)
    return hud_frame


def _draw_input_hud(frame: RgbFrame, info: Mapping[str, object]) -> None:
    height, width = frame.shape[:2]
    scale = max(1, min(width, height) // 220)
    margin = 2 * scale
    hud_width = 34 * scale
    hud_height = 49 * scale
    x = width - hud_width - margin
    y = min(height - hud_height - margin, max(margin, int(height * 0.23)))

    stick_size = 24 * scale
    stick_x = x + (hud_width - stick_size) // 2
    stick_y = y
    _draw_stick(
        frame,
        stick_x,
        stick_y,
        size=stick_size,
        stick_x_value=_float_info(info, "watch_recording_input_stick_x"),
        pitch_value=_float_info(info, "watch_recording_input_pitch"),
    )

    row_x = x + (hud_width - 25 * scale) // 2
    button_radius = 5 * scale
    lr_y = y + stick_size + 4 * scale
    _draw_round_button(
        frame,
        "Z",
        center_x=row_x + button_radius,
        center_y=lr_y + button_radius,
        radius=button_radius,
        active=_bool_info(info, "watch_recording_input_lean_left"),
    )
    _draw_round_button(
        frame,
        "R",
        center_x=row_x + 15 * scale + button_radius,
        center_y=lr_y + button_radius,
        radius=button_radius,
        active=_bool_info(info, "watch_recording_input_lean_right"),
    )

    ab_y = lr_y + 13 * scale
    _draw_round_button(
        frame,
        "A",
        center_x=row_x + button_radius,
        center_y=ab_y + button_radius,
        radius=button_radius,
        active=_bool_info(info, "watch_recording_input_gas"),
    )
    _draw_round_button(
        frame,
        "B",
        center_x=row_x + 15 * scale + button_radius,
        center_y=ab_y + button_radius,
        radius=button_radius,
        active=_bool_info(info, "watch_recording_input_boost"),
    )


def _draw_stick(
    frame: RgbFrame,
    x: int,
    y: int,
    *,
    size: int,
    stick_x_value: float,
    pitch_value: float,
) -> None:
    _blend_rect(frame, x + 1, y + 1, size, size, (0, 0, 0), alpha=0.25)
    _blend_rect(frame, x, y, size, size, (6, 12, 22), alpha=0.44)
    _draw_box_outline(frame, x, y, size, size, (57, 92, 110))
    center_x = x + size // 2
    center_y = y + size // 2
    _draw_rect(frame, x + 4, center_y, size - 8, 1, (40, 69, 83))
    _draw_rect(frame, center_x, y + 4, 1, size - 8, (40, 69, 83))
    radius = max(1, size // 2 - 4)
    puck_x = center_x + int(_clamp(stick_x_value, -1.0, 1.0) * radius)
    puck_y = center_y + int(_clamp(pitch_value, -1.0, 1.0) * radius)
    _draw_crosshair(frame, puck_x, puck_y, size=max(3, size // 6), color=(129, 202, 255))


def _draw_round_button(
    frame: RgbFrame,
    label: str,
    *,
    center_x: int,
    center_y: int,
    radius: int,
    active: bool,
) -> None:
    fill = (74, 216, 142) if active else (18, 27, 38)
    text = (3, 20, 13) if active else (225, 239, 245)
    _blend_circle(frame, center_x + 1, center_y + 1, radius, (0, 0, 0), alpha=0.24)
    _blend_circle(frame, center_x, center_y, radius, fill, alpha=0.84 if active else 0.58)
    _draw_circle_ring(
        frame,
        center_x,
        center_y,
        radius,
        (111, 255, 180) if active else (42, 64, 76),
        thickness=max(1, radius // 5),
        alpha=0.78 if active else 0.62,
    )
    _blend_circle(
        frame,
        center_x,
        center_y,
        max(1, radius - 2),
        fill,
        alpha=0.18 if active else 0.08,
    )
    text_scale = max(1, radius // 6)
    text_width = _tiny_text_width(label, scale=text_scale)
    text_x = center_x - text_width // 2
    text_y = center_y - (_tiny_text_height(label, scale=text_scale) // 2)
    _draw_tiny_text(frame, label, text_x, text_y, scale=text_scale, color=text, bold=True)


def _draw_tiny_text(
    frame: RgbFrame,
    text: str,
    x: int,
    y: int,
    *,
    scale: int,
    color: _RgbColor,
    bold: bool = False,
) -> None:
    cursor_x = x
    for char in text.upper():
        glyph = _HUD_TINY_GLYPHS.get(char)
        if glyph is None:
            cursor_x += 4 * scale
            continue
        pixel_width = scale + (max(1, scale // 2) if bold else 0)
        for row_index, row in enumerate(glyph):
            for column_index, pixel in enumerate(row):
                if pixel == "1":
                    _draw_rect(
                        frame,
                        cursor_x + column_index * scale,
                        y + row_index * scale,
                        pixel_width,
                        scale,
                        color,
                    )
        cursor_x += (len(glyph[0]) + 1) * scale


def _tiny_text_width(text: str, *, scale: int) -> int:
    width = 0
    for char in text.upper():
        glyph = _HUD_TINY_GLYPHS.get(char)
        width += (len(glyph[0]) + 1 if glyph is not None else 4) * scale
    return max(0, width - scale)


def _tiny_text_height(text: str, *, scale: int) -> int:
    heights = [len(glyph) for char in text.upper() if (glyph := _HUD_TINY_GLYPHS.get(char))]
    return (max(heights) if heights else 0) * scale


def _draw_box_outline(
    frame: RgbFrame, x: int, y: int, width: int, height: int, color: _RgbColor
) -> None:
    _draw_rect(frame, x, y, width, 1, color)
    _draw_rect(frame, x, y + height - 1, width, 1, color)
    _draw_rect(frame, x, y, 1, height, color)
    _draw_rect(frame, x + width - 1, y, 1, height, color)


def _draw_rect(frame: RgbFrame, x: int, y: int, width: int, height: int, color: _RgbColor) -> None:
    x0, y0, x1, y1 = _clipped_rect(frame, x, y, width, height)
    if x0 >= x1 or y0 >= y1:
        return
    frame[y0:y1, x0:x1] = color


def _blend_rect(
    frame: RgbFrame,
    x: int,
    y: int,
    width: int,
    height: int,
    color: _RgbColor,
    *,
    alpha: float,
) -> None:
    x0, y0, x1, y1 = _clipped_rect(frame, x, y, width, height)
    if x0 >= x1 or y0 >= y1:
        return
    region = frame[y0:y1, x0:x1]
    color_array = np.asarray(color, dtype=np.float32)
    region[:] = (region.astype(np.float32) * (1.0 - alpha) + color_array * alpha).astype(np.uint8)


def _blend_circle(
    frame: RgbFrame,
    center_x: int,
    center_y: int,
    radius: int,
    color: _RgbColor,
    *,
    alpha: float,
) -> None:
    if radius <= 0:
        return
    frame_height, frame_width = frame.shape[:2]
    color_array = np.asarray(color, dtype=np.float32)
    radius_squared = radius * radius
    y0 = max(0, center_y - radius)
    y1 = min(frame_height, center_y + radius + 1)
    for row_y in range(y0, y1):
        row_offset = row_y - center_y
        half_width = int((radius_squared - row_offset * row_offset) ** 0.5)
        x0 = max(0, center_x - half_width)
        x1 = min(frame_width, center_x + half_width + 1)
        if x0 >= x1:
            continue
        region = frame[row_y : row_y + 1, x0:x1]
        region[:] = (region.astype(np.float32) * (1.0 - alpha) + color_array * alpha).astype(
            np.uint8
        )


def _draw_circle_ring(
    frame: RgbFrame,
    center_x: int,
    center_y: int,
    radius: int,
    color: _RgbColor,
    *,
    thickness: int,
    alpha: float,
) -> None:
    if radius <= 0 or thickness <= 0:
        return
    frame_height, frame_width = frame.shape[:2]
    color_array = np.asarray(color, dtype=np.float32)
    outer_squared = radius * radius
    inner_radius = max(0, radius - thickness)
    inner_squared = inner_radius * inner_radius
    y0 = max(0, center_y - radius)
    y1 = min(frame_height, center_y + radius + 1)
    for row_y in range(y0, y1):
        row_offset = row_y - center_y
        x0 = max(0, center_x - radius)
        x1 = min(frame_width, center_x + radius + 1)
        for pixel_x in range(x0, x1):
            column_offset = pixel_x - center_x
            distance_squared = column_offset * column_offset + row_offset * row_offset
            if inner_squared < distance_squared <= outer_squared:
                pixel = frame[row_y, pixel_x]
                frame[row_y, pixel_x] = (
                    pixel.astype(np.float32) * (1.0 - alpha) + color_array * alpha
                ).astype(np.uint8)


def _draw_crosshair(
    frame: RgbFrame,
    center_x: int,
    center_y: int,
    *,
    size: int,
    color: _RgbColor,
) -> None:
    half = max(2, size // 2)
    gap = max(1, size // 5)
    _draw_rect(frame, center_x - half, center_y, half - gap + 1, 1, color)
    _draw_rect(frame, center_x + gap, center_y, half - gap + 1, 1, color)
    _draw_rect(frame, center_x, center_y - half, 1, half - gap + 1, color)
    _draw_rect(frame, center_x, center_y + gap, 1, half - gap + 1, color)
    _draw_rect(frame, center_x, center_y, 1, 1, (224, 242, 255))


def _clipped_rect(
    frame: RgbFrame,
    x: int,
    y: int,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    frame_height, frame_width = frame.shape[:2]
    return (
        max(0, x),
        max(0, y),
        min(frame_width, x + width),
        min(frame_height, y + height),
    )


def _bool_info(info: Mapping[str, object], key: str) -> bool:
    return info.get(key) is True


def _float_info(info: Mapping[str, object], key: str) -> float:
    value = info.get(key)
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return 0.0


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return min(max(value, minimum), maximum)
