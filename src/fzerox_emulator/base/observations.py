# src/fzerox_emulator/base/observations.py
"""Observation value types plus lazy native geometry helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias, TypedDict

ObservationStackMode: TypeAlias = Literal["rgb", "gray", "luma_chroma"]
ObservationResizeFilter: TypeAlias = Literal["nearest", "bilinear"]


class FrameObservationOptions(TypedDict, total=False):
    """Typed Python-side options passed into the native observation renderer."""

    stack_mode: ObservationStackMode
    minimap_layer: bool
    resize_filter: ObservationResizeFilter
    minimap_resize_filter: ObservationResizeFilter
    height: int
    width: int


@dataclass(frozen=True)
class ObservationSpec:
    """Resolved native observation geometry for one image layout."""

    preset: str
    width: int
    height: int
    channels: int
    display_width: int
    display_height: int


@dataclass(frozen=True, slots=True)
class ObservationImageRecipe:
    """One native image-render recipe for an observation view."""

    preset: str | None = None
    height: int | None = None
    width: int | None = None
    frame_stack: int = 1
    stack_mode: ObservationStackMode = "rgb"
    minimap_layer: bool = False
    resize_filter: ObservationResizeFilter = "nearest"
    minimap_resize_filter: ObservationResizeFilter = "nearest"


def stacked_observation_channels(
    single_frame_channels: int,
    *,
    frame_stack: int,
    stack_mode: ObservationStackMode,
    minimap_layer: bool = False,
) -> int:
    """Return channel count after temporal frame-stack encoding."""

    # Import lazily so `fzerox_emulator.base` stays usable without loading the
    # compiled extension.
    import fzerox_emulator._native as _native

    return int(
        _native.stacked_observation_channels(
            single_frame_channels,
            frame_stack,
            stack_mode,
            minimap_layer,
        )
    )


def display_size(frame_shape: tuple[int, ...], aspect_ratio: float) -> tuple[int, int]:
    """Convert a raw frame shape into a human display size."""

    if len(frame_shape) < 2:
        raise ValueError("Frame shape must include height and width")

    frame_height, frame_width = frame_shape[:2]
    # Keep the aspect-ratio math in Rust/native as the single source of truth.
    import fzerox_emulator._native as _native

    display_width, display_height = _native.display_size(
        int(frame_width),
        int(frame_height),
        float(aspect_ratio),
    )
    return int(display_width), int(display_height)
