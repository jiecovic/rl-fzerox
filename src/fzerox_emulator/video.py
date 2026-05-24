# src/fzerox_emulator/video.py
from __future__ import annotations


def display_size(frame_shape: tuple[int, ...], aspect_ratio: float) -> tuple[int, int]:
    """Convert a raw frame shape into a human display size."""

    if len(frame_shape) < 2:
        raise ValueError("Frame shape must include height and width")

    frame_height, frame_width = frame_shape[:2]
    import fzerox_emulator._native as _native

    display_width, display_height = _native.display_size(
        int(frame_width),
        int(frame_height),
        float(aspect_ratio),
    )
    return int(display_width), int(display_height)
