# src/rl_fzerox/core/runtime_spec/schema/observations.py
"""Observation runtime schema and state-component validation.

Image geometry, frame stacking, minimap settings, and optional scalar state
components are validated here before env code builds observation spaces. The
state-component feature checks deliberately call into env observation code so
invalid feature selections fail at config load time.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Literal, TypedDict

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveInt,
    field_validator,
    model_serializer,
    model_validator,
)

from rl_fzerox.core.domain.observations import (
    OBSERVATION_IMAGE_GEOMETRY,
    ActionHistoryControlName,
    ObservationCourseContextName,
    ObservationPresetName,
    ObservationRendererName,
    ObservationStateComponentName,
    ObservationStateComponentSettings,
    TrackPositionProgressSourceName,
    preset_geometry,
    source_crop_geometry,
)
from rl_fzerox.core.runtime_spec.renderers import RendererName
from rl_fzerox.core.runtime_spec.schema.common import (
    ObservationResizeFilter,
)

type ObservationResolutionMode = Literal["preset", "custom", "source_crop"]


class PresetResolutionChoice(BaseModel):
    """Preset-backed observation resolution."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["preset"] = "preset"
    preset: ObservationPresetName = OBSERVATION_IMAGE_GEOMETRY.default_preset


class CustomResolutionChoice(BaseModel):
    """Custom fixed observation resolution."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["custom"] = "custom"
    height: int = Field(
        ge=OBSERVATION_IMAGE_GEOMETRY.custom_bounds.min_dimension,
        le=OBSERVATION_IMAGE_GEOMETRY.custom_bounds.max_height,
    )
    width: int = Field(
        ge=OBSERVATION_IMAGE_GEOMETRY.custom_bounds.min_dimension,
        le=OBSERVATION_IMAGE_GEOMETRY.custom_bounds.max_width,
    )


class SourceCropResolutionChoice(BaseModel):
    """Use the renderer-native crop size without target downsampling."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["source_crop"] = "source_crop"


type ObservationResolutionConfig = Annotated[
    PresetResolutionChoice | CustomResolutionChoice | SourceCropResolutionChoice,
    Field(discriminator="mode"),
]


def resolve_observation_geometry(
    *,
    resolution: ObservationResolutionConfig,
    renderer: ObservationRendererName | None = None,
) -> tuple[int, int]:
    """Return the active `(height, width)` pair for one observation config."""

    if isinstance(resolution, PresetResolutionChoice):
        return preset_geometry(resolution.preset)
    if isinstance(resolution, CustomResolutionChoice):
        return int(resolution.height), int(resolution.width)
    if renderer is None:
        raise ValueError("renderer must be set for source-crop observation resolution")
    return source_crop_geometry(renderer)


class ObservationStateComponentConfig(BaseModel):
    """One ordered scalar-state component in the image-state observation."""

    model_config = ConfigDict(extra="forbid")

    name: ObservationStateComponentName
    encoding: ObservationCourseContextName | None = None
    progress_source: TrackPositionProgressSourceName | None = None
    length: PositiveInt | None = Field(default=None, le=16)
    controls: tuple[ActionHistoryControlName, ...] | None = None
    included_features: tuple[str, ...] | None = None

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
                "included_features",
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
        if self.included_features is not None:
            if len(set(self.included_features)) != len(self.included_features):
                raise ValueError("state component included_features must not contain duplicates")
            supported = set(_supported_state_feature_names(self))
            unsupported = sorted(set(self.included_features) - supported)
            if unsupported:
                joined = ", ".join(unsupported)
                raise ValueError(f"{self.name} does not support included feature(s): {joined}")
        return self

    def _allowed_fields(self) -> frozenset[str]:
        match self.name:
            case "course_context":
                return frozenset({"encoding", "included_features"})
            case "control_history":
                return frozenset({"length", "controls", "included_features"})
            case "track_position":
                return frozenset({"progress_source", "included_features"})
            case "vehicle_state" | "machine_context" | "surface_state":
                return frozenset({"included_features"})
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
            included_features=self.included_features,
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
    resolution: ObservationResolutionConfig = Field(default_factory=PresetResolutionChoice)
    frame_stack: PositiveInt = 4
    stack_mode: Literal["rgb", "gray", "luma_chroma"] = "rgb"
    minimap_layer: bool = False
    resize_filter: ObservationResizeFilter = "nearest"
    minimap_resize_filter: ObservationResizeFilter = "nearest"
    state_components: tuple[ObservationStateComponentConfig, ...] | None = None

    @model_serializer(mode="plain")
    def _serialize(self) -> dict[str, object]:
        data: dict[str, object] = {
            "mode": self.mode,
            "resolution": self.resolution.model_dump(mode="json"),
            "frame_stack": self.frame_stack,
            "stack_mode": self.stack_mode,
            "minimap_layer": self.minimap_layer,
            "resize_filter": self.resize_filter,
            "minimap_resize_filter": self.minimap_resize_filter,
        }
        if self.state_components is not None:
            data["state_components"] = [
                component.model_dump(mode="json") for component in self.state_components
            ]
        return data

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
        if self.mode == "image_state" and not self.state_components:
            raise ValueError("observation.state_components must not be empty for mode=image_state")
        return self

    def state_components_data(self) -> tuple[ObservationStateComponentSettings, ...] | None:
        """Return state-component settings in the plain form consumed by env code."""

        if self.state_components is None:
            return None
        return tuple(component.data() for component in self.state_components)

    def image_geometry(self, *, renderer: RendererName | None = None) -> tuple[int, int]:
        """Return the active `(height, width)` for one resolved runtime config."""

        return resolve_observation_geometry(
            resolution=self.resolution,
            renderer=renderer,
        )

    def native_resolution_kwargs(
        self,
        *,
        renderer: RendererName | None = None,
    ) -> NativeObservationResolutionKwargs:
        """Return the backend kwargs for the active preset-or-custom resolution."""

        if isinstance(self.resolution, PresetResolutionChoice):
            return {"preset": self.resolution.preset}
        height, width = self.image_geometry(renderer=renderer)
        return {"height": height, "width": width}


def _supported_state_feature_names(component: ObservationStateComponentConfig) -> tuple[str, ...]:
    from rl_fzerox.core.envs.observations.state.components import raw_state_component_features

    settings = _component_settings(component, included_features=None)
    feature_names = [
        feature.name
        for feature in raw_state_component_features(
            settings,
            split_lean_history=False,
        )
    ]
    if component.name == "control_history":
        for feature in raw_state_component_features(settings, split_lean_history=True):
            if feature.name not in feature_names:
                feature_names.append(feature.name)
    return tuple(feature_names)


def _component_settings(
    component: ObservationStateComponentConfig,
    *,
    included_features: tuple[str, ...] | None,
) -> ObservationStateComponentSettings:
    return ObservationStateComponentSettings(
        name=component.name,
        encoding=component.encoding,
        progress_source=component.progress_source,
        length=None if component.length is None else int(component.length),
        controls=component.controls,
        included_features=included_features,
    )
