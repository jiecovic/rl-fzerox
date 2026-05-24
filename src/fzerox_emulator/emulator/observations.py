# src/fzerox_emulator/emulator/observations.py
"""Observation rendering methods mixed into the concrete emulator wrapper."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fzerox_emulator.arrays import ObservationFrame, RgbFrame
from fzerox_emulator.base.observations import (
    FrameObservationOptions,
    ObservationResizeFilter,
    ObservationSpec,
    ObservationStackMode,
    display_size,
)

if TYPE_CHECKING:
    from fzerox_emulator._native import Emulator as NativeEmulator


class ObservationRenderingMixin:
    """Methods that convert native frames into Python-facing frame objects."""

    _native: NativeEmulator
    _observation_specs: dict[tuple[str | None, int | None, int | None], ObservationSpec]

    @property
    def display_size(self) -> tuple[int, int]:
        """Return the aspect-corrected display size for the latest frame shape."""

        return display_size(self._frame_shape(), self._display_aspect_ratio())

    def render(self) -> RgbFrame:
        """Return the latest raw RGB frame as a NumPy array."""

        return self._native.frame_rgb()

    def render_display(
        self,
        *,
        preset: str | None = None,
        height: int | None = None,
        width: int | None = None,
    ) -> RgbFrame:
        """Return one native display frame for the requested image layout."""

        return self._native.frame_display(
            "" if preset is None else preset,
            height=height,
            width=width,
        )

    def observation_spec(
        self,
        preset: str | None = None,
        *,
        height: int | None = None,
        width: int | None = None,
    ) -> ObservationSpec:
        """Return the resolved native observation spec for one image layout."""

        cache_key = (preset, height, width)
        cached = self._observation_specs.get(cache_key)
        if cached is not None:
            return cached
        spec_data = self._native.observation_spec(
            "" if preset is None else preset,
            height=height,
            width=width,
        )
        spec = ObservationSpec(
            preset=str(spec_data["preset"]),
            width=int(spec_data["width"]),
            height=int(spec_data["height"]),
            channels=int(spec_data["channels"]),
            display_width=int(spec_data["display_width"]),
            display_height=int(spec_data["display_height"]),
        )
        self._observation_specs[cache_key] = spec
        return spec

    def render_observation(
        self,
        *,
        preset: str | None = None,
        height: int | None = None,
        width: int | None = None,
        frame_stack: int,
        stack_mode: ObservationStackMode = "rgb",
        minimap_layer: bool = False,
        resize_filter: ObservationResizeFilter = "nearest",
        minimap_resize_filter: ObservationResizeFilter = "nearest",
    ) -> ObservationFrame:
        """Return one native stacked observation tensor for the requested image layout."""

        options: FrameObservationOptions = {
            "stack_mode": stack_mode,
            "minimap_layer": minimap_layer,
            "resize_filter": resize_filter,
            "minimap_resize_filter": minimap_resize_filter,
        }
        if height is not None:
            options["height"] = height
        if width is not None:
            options["width"] = width
        return self._native.frame_observation(
            "" if preset is None else preset,
            frame_stack,
            options,
        )

    def _frame_shape(self) -> tuple[int, int, int]:
        height, width, channels = self._native.frame_shape
        return int(height), int(width), int(channels)

    def _display_aspect_ratio(self) -> float:
        return float(self._native.display_aspect_ratio)
