# src/rl_fzerox/core/runtime_spec/schema/observations.py
from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, field_validator, model_validator

from rl_fzerox.core.domain.observation_components import (
    ActionHistoryControlName,
    ObservationCourseContextName,
    ObservationStateComponentName,
    ObservationStateComponentSettings,
    TrackPositionProgressSourceName,
)
from rl_fzerox.core.domain.observation_image import (
    ObservationCustomResolution,
    ObservationResolutionMode,
    resolve_observation_geometry,
)
from rl_fzerox.core.runtime_spec.schema.common import (
    ObservationPresetName,
    ObservationResizeFilter,
)


class ObservationStateComponentConfig(BaseModel):
    """One ordered scalar-state component in the image-state observation."""

    model_config = ConfigDict(extra="forbid")

    name: ObservationStateComponentName
    encoding: ObservationCourseContextName | None = None
    progress_source: TrackPositionProgressSourceName | None = None
    length: PositiveInt | None = Field(default=None, le=16)
    controls: tuple[ActionHistoryControlName, ...] | None = None

    @model_validator(mode="before")
    @classmethod
    def _parse_lego_component(cls, data: object) -> object:
        if isinstance(data, str):
            return {"name": data}
        if not isinstance(data, Mapping) or "name" in data or len(data) != 1:
            return data

        name, settings = next(iter(data.items()))
        if settings is None:
            return {"name": name}
        if isinstance(settings, Mapping):
            return {"name": name, **settings}
        return data

    @model_validator(mode="after")
    def _validate_component_settings(self) -> ObservationStateComponentConfig:
        configured_fields = {
            name
            for name in (
                "encoding",
                "progress_source",
                "length",
                "controls",
            )
            if getattr(self, name) is not None
        }
        invalid_fields = configured_fields - self._allowed_fields()
        if invalid_fields:
            joined = ", ".join(sorted(invalid_fields))
            raise ValueError(f"{self.name} does not accept setting(s): {joined}")

        if self.controls is not None:
            if len(set(self.controls)) != len(self.controls):
                raise ValueError("control_history.controls must not contain duplicates")
            normalized = {"gas" if control == "thrust" else control for control in self.controls}
            if len(normalized) != len(self.controls):
                raise ValueError("control_history.controls cannot contain both gas and thrust")
        return self

    def _allowed_fields(self) -> frozenset[str]:
        match self.name:
            case "course_context":
                return frozenset({"encoding"})
            case "control_history":
                return frozenset({"length", "controls"})
            case "track_position":
                return frozenset({"progress_source"})
            case "vehicle_state" | "machine_context" | "surface_state":
                return frozenset()
            case _:
                raise ValueError(f"Unsupported state component: {self.name!r}")

    def data(self) -> ObservationStateComponentSettings:
        """Return the compact ordered form consumed by env code."""

        return ObservationStateComponentSettings(
            name=self.name,
            encoding=self.encoding,
            progress_source=self.progress_source,
            length=None if self.length is None else int(self.length),
            controls=self.controls,
        )


class NativeObservationResolutionKwargs(TypedDict, total=False):
    """Typed backend kwargs for preset-based or custom observation layouts."""

    preset: str
    height: int
    width: int


class ObservationConfig(BaseModel):
    """Observation adapter settings for the current env."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["image", "image_state"] = "image"
    resolution_mode: ObservationResolutionMode = "preset"
    preset: ObservationPresetName = "crop_60x76"
    custom_resolution: ObservationCustomResolution | None = None
    frame_stack: PositiveInt = 4
    stack_mode: Literal["rgb", "gray", "luma_chroma"] = "rgb"
    minimap_layer: bool = False
    resize_filter: ObservationResizeFilter = "nearest"
    minimap_resize_filter: ObservationResizeFilter = "nearest"
    state_components: tuple[ObservationStateComponentConfig, ...] | None = None

    @field_validator("state_components")
    @classmethod
    def _validate_unique_state_components(
        cls,
        value: tuple[ObservationStateComponentConfig, ...] | None,
    ) -> tuple[ObservationStateComponentConfig, ...] | None:
        if value is None:
            return None
        names = [component.name for component in value]
        if len(set(names)) != len(names):
            raise ValueError("observation.state_components must not contain duplicates")
        return value

    @model_validator(mode="after")
    def _validate_state_components_for_mode(self) -> ObservationConfig:
        if self.resolution_mode == "preset" and self.custom_resolution is not None:
            raise ValueError(
                "observation.custom_resolution must be null for resolution_mode='preset'"
            )
        if self.resolution_mode == "custom" and self.custom_resolution is None:
            raise ValueError(
                "observation.custom_resolution must be set for resolution_mode='custom'"
            )
        if self.mode == "image_state" and not self.state_components:
            raise ValueError("observation.state_components must not be empty for mode=image_state")
        return self

    def state_components_data(self) -> tuple[ObservationStateComponentSettings, ...] | None:
        """Return state-component settings in the plain form consumed by env code."""

        if self.state_components is None:
            return None
        return tuple(component.data() for component in self.state_components)

    def image_geometry(self) -> tuple[int, int]:
        """Return the active `(height, width)` for one resolved runtime config."""

        return resolve_observation_geometry(
            resolution_mode=self.resolution_mode,
            preset=self.preset,
            custom_resolution=self.custom_resolution,
        )

    def native_resolution_kwargs(self) -> NativeObservationResolutionKwargs:
        """Return the backend kwargs for the active preset-or-custom resolution."""

        if self.resolution_mode == "preset":
            return {"preset": self.preset}
        height, width = self.image_geometry()
        return {"height": height, "width": width}
