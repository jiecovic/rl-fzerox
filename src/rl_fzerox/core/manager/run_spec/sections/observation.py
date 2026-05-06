# src/rl_fzerox/core/manager/run_spec/sections/observation.py
"""Observation section of the manager-owned run-spec model."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator

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
from rl_fzerox.core.manager.run_spec.common import (
    ObservationPreset,
    ObservationResizeFilter,
    StackMode,
)


class ManagedStateComponentConfig(BaseModel):
    """One state-vector component exposed by the run manager."""

    model_config = ConfigDict(extra="forbid")

    name: ObservationStateComponentName
    encoding: ObservationCourseContextName | None = None
    progress_source: TrackPositionProgressSourceName | None = None
    length: PositiveInt | None = Field(default=None, le=16)
    controls: tuple[ActionHistoryControlName, ...] | None = None

    @model_validator(mode="after")
    def _validate_component_settings(self) -> ManagedStateComponentConfig:
        configured_fields = {
            name
            for name in ("encoding", "progress_source", "length", "controls")
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
        return ObservationStateComponentSettings(
            name=self.name,
            encoding=self.encoding,
            progress_source=self.progress_source,
            length=None if self.length is None else int(self.length),
            controls=self.controls,
        )


class ManagedStateFeatureDropoutConfig(BaseModel):
    """Episode-scoped dropout override for one concrete scalar state feature."""

    model_config = ConfigDict(extra="forbid")

    name: str
    dropout_prob: float = Field(default=0.0, ge=0.0, le=1.0)


class ManagedObservationConfig(BaseModel):
    """Observation knobs exposed by the run manager."""

    model_config = ConfigDict(extra="forbid")

    resolution_mode: ObservationResolutionMode = "preset"
    preset: ObservationPreset = "crop_60x76"
    custom_resolution: ObservationCustomResolution | None = None
    frame_stack: PositiveInt = Field(default=2, le=8)
    stack_mode: StackMode = "rgb"
    minimap_layer: bool = False
    resize_filter: ObservationResizeFilter = "bilinear"
    minimap_resize_filter: ObservationResizeFilter = "nearest"
    state_components: tuple[ManagedStateComponentConfig, ...] = Field(
        default_factory=lambda: default_state_components()
    )
    state_feature_dropouts: tuple[ManagedStateFeatureDropoutConfig, ...] = Field(
        default_factory=lambda: default_state_feature_dropouts()
    )

    @model_validator(mode="after")
    def _validate_observation_components(self) -> ManagedObservationConfig:
        if self.resolution_mode == "preset" and self.custom_resolution is not None:
            raise ValueError(
                "observation.custom_resolution must be null for resolution_mode='preset'"
            )
        if self.resolution_mode == "custom" and self.custom_resolution is None:
            raise ValueError(
                "observation.custom_resolution must be set for resolution_mode='custom'"
            )
        names = [component.name for component in self.state_components]
        if len(set(names)) != len(names):
            raise ValueError("observation.state_components must not contain duplicates")
        feature_names = [feature.name for feature in self.state_feature_dropouts]
        if len(set(feature_names)) != len(feature_names):
            raise ValueError("observation.state_feature_dropouts must not contain duplicates")
        return self

    def image_geometry(self) -> tuple[int, int]:
        """Return the active `(height, width)` for preview and projection code."""

        return resolve_observation_geometry(
            resolution_mode=self.resolution_mode,
            preset=self.preset,
            custom_resolution=self.custom_resolution,
        )


DEFAULT_STATE_COMPONENTS: tuple[ManagedStateComponentConfig, ...] = (
    ManagedStateComponentConfig(name="vehicle_state"),
    ManagedStateComponentConfig(name="machine_context"),
    ManagedStateComponentConfig(name="track_position", progress_source="segment_progress"),
    ManagedStateComponentConfig(name="surface_state"),
    ManagedStateComponentConfig(name="course_context", encoding="one_hot_builtin"),
    ManagedStateComponentConfig(
        name="control_history",
        length=1,
        controls=("steer", "thrust", "air_brake", "boost", "lean", "pitch"),
    ),
)
DEFAULT_STATE_FEATURE_DROPOUTS: tuple[ManagedStateFeatureDropoutConfig, ...] = (
    ManagedStateFeatureDropoutConfig(name="track_position.edge_ratio", dropout_prob=1.0),
    ManagedStateFeatureDropoutConfig(
        name="track_position.outside_track_bounds",
        dropout_prob=1.0,
    ),
)


def default_state_components() -> tuple[ManagedStateComponentConfig, ...]:
    """Return fresh state-component config objects for manager defaults."""

    return tuple(component.model_copy(deep=True) for component in DEFAULT_STATE_COMPONENTS)


def default_state_feature_dropouts() -> tuple[ManagedStateFeatureDropoutConfig, ...]:
    """Return feature-level state dropouts that preserve the current run shape."""

    return tuple(feature.model_copy(deep=True) for feature in DEFAULT_STATE_FEATURE_DROPOUTS)


def managed_state_component_feature_names(
    components: tuple[ManagedStateComponentConfig, ...],
    *,
    independent_lean_buttons: bool = False,
) -> frozenset[str]:
    from rl_fzerox.core.envs.observations.state.components import state_component_features

    names: set[str] = set()
    for component in components:
        settings = component.data()
        for feature in state_component_features(
            settings,
            independent_lean_buttons=independent_lean_buttons,
        ):
            names.add(feature.name)
    return frozenset(names)
