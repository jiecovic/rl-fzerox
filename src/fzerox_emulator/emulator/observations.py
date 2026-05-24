# src/fzerox_emulator/emulator/observations.py
from __future__ import annotations

from fzerox_emulator._native import Emulator as NativeEmulator
from fzerox_emulator.arrays import ObservationFrame, RgbFrame
from fzerox_emulator.base import (
    FrameObservationOptions,
    ObservationResizeFilter,
    ObservationSpec,
    ObservationStackMode,
    normalize_observation_resolution,
)
from fzerox_emulator.frames import (
    expected_observation_shape,
    validated_display_frame,
    validated_observation_frame,
    validated_raw_rgb_frame,
)
from fzerox_emulator.video import display_size


class ObservationRenderingMixin:
    _native: NativeEmulator
    _observation_specs: dict[tuple[str | None, int | None, int | None], ObservationSpec]

    @property
    def display_size(self) -> tuple[int, int]:
        return display_size(self._frame_shape(), self._display_aspect_ratio())

    def render(self) -> RgbFrame:
        """Return the latest raw RGB frame as a NumPy array."""

        return validated_raw_rgb_frame(self._native.frame_rgb(), self._frame_shape())

    def render_display(
        self,
        *,
        preset: str | None = None,
        height: int | None = None,
        width: int | None = None,
    ) -> RgbFrame:
        """Return one native display frame for the requested image layout."""

        resolved_preset, resolved_height, resolved_width = normalize_observation_resolution(
            preset=preset,
            height=height,
            width=width,
        )
        spec = self.observation_spec(resolved_preset, height=resolved_height, width=resolved_width)
        expected_shape = (spec.display_height, spec.display_width, 3)
        return validated_display_frame(
            self._native.frame_display(
                "" if resolved_preset is None else resolved_preset,
                height=resolved_height,
                width=resolved_width,
            ),
            expected_shape=expected_shape,
        )

    def observation_spec(
        self,
        preset: str | None = None,
        *,
        height: int | None = None,
        width: int | None = None,
    ) -> ObservationSpec:
        """Return the resolved native observation spec for one image layout."""

        resolved_preset, resolved_height, resolved_width = normalize_observation_resolution(
            preset=preset,
            height=height,
            width=width,
        )
        cache_key = (resolved_preset, resolved_height, resolved_width)
        cached = self._observation_specs.get(cache_key)
        if cached is not None:
            return cached
        spec_data = self._native.observation_spec(
            "" if resolved_preset is None else resolved_preset,
            height=resolved_height,
            width=resolved_width,
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

        resolved_preset, resolved_height, resolved_width = normalize_observation_resolution(
            preset=preset,
            height=height,
            width=width,
        )
        spec = self.observation_spec(
            resolved_preset,
            height=resolved_height,
            width=resolved_width,
        )
        options: FrameObservationOptions = {
            "stack_mode": stack_mode,
            "minimap_layer": minimap_layer,
            "resize_filter": resize_filter,
            "minimap_resize_filter": minimap_resize_filter,
        }
        if resolved_height is not None:
            options["height"] = resolved_height
        if resolved_width is not None:
            options["width"] = resolved_width
        frame = self._native.frame_observation(
            "" if resolved_preset is None else resolved_preset,
            frame_stack,
            options,
        )
        expected_shape = expected_observation_shape(
            spec,
            frame_stack=frame_stack,
            stack_mode=stack_mode,
            minimap_layer=minimap_layer,
        )
        return validated_observation_frame(
            frame,
            expected_shape=expected_shape,
            source_label="native emulator",
        )

    def _frame_shape(self) -> tuple[int, int, int]:
        height, width, channels = self._native.frame_shape
        return int(height), int(width), int(channels)

    def _display_aspect_ratio(self) -> float:
        return float(self._native.display_aspect_ratio)
