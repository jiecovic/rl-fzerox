# src/rl_fzerox/core/emulator/video.py
from __future__ import annotations


def display_size(frame_shape: tuple[int, ...], aspect_ratio: float) -> tuple[int, int]:
    """Convert a raw frame shape into a human display size."""

    if len(frame_shape) < 2:
        raise ValueError("Frame shape must include height and width")

    frame_height, frame_width = frame_shape[:2]
    if aspect_ratio <= 0.0:
        return frame_width, frame_height

    display_height = max(1, round(frame_width / aspect_ratio))
    return frame_width, int(display_height)
