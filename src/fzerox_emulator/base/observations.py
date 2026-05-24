# src/fzerox_emulator/base/observations.py
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

    def normalized_resolution(self) -> tuple[str | None, int | None, int | None]:
        return normalize_observation_resolution(
            preset=self.preset,
            height=self.height,
            width=self.width,
        )


def normalize_observation_resolution(
    *,
    preset: str | None = None,
    height: int | None = None,
    width: int | None = None,
) -> tuple[str | None, int | None, int | None]:
    """Return one validated preset-or-custom resolution triple."""

    if preset is not None:
        if height is not None or width is not None:
            raise ValueError("preset cannot be combined with custom observation height/width")
        return preset, None, None
    if height is None or width is None:
        raise ValueError("custom observation height and width must both be set")
    return None, int(height), int(width)


def stacked_observation_channels(
    single_frame_channels: int,
    *,
    frame_stack: int,
    stack_mode: ObservationStackMode,
    minimap_layer: bool = False,
) -> int:
    """Return channel count after temporal frame-stack encoding."""

    if frame_stack <= 0:
        raise ValueError("frame_stack must be positive")
    extra_channels = 1 if minimap_layer else 0
    if stack_mode == "rgb":
        return (single_frame_channels * frame_stack) + extra_channels
    if stack_mode == "gray":
        return frame_stack + extra_channels
    if stack_mode == "luma_chroma":
        return (frame_stack * 2) + extra_channels
    raise ValueError(f"Unsupported observation stack mode: {stack_mode!r}")
