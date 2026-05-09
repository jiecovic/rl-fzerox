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
    included_features: tuple[str, ...] | None = None
    optional_features: tuple[str, ...] | None = Field(default=None, exclude=True, repr=False)

    @model_validator(mode="after")
    def _validate_component_settings(self) -> ManagedStateComponentConfig:
        self._normalize_legacy_optional_features()
        configured_fields = {
            name
            for name in (
                "encoding",
                "progress_source",
                "length",
                "controls",
                "included_features",
                "optional_features",
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

    def _normalize_legacy_optional_features(self) -> None:
        if self.optional_features is None:
            return
        if self.included_features is not None:
            raise ValueError(
                "state component cannot define both included_features and optional_features"
            )
        if len(set(self.optional_features)) != len(self.optional_features):
            raise ValueError("state component optional_features must not contain duplicates")
        supported_names = _supported_state_feature_names(self)
        default_names = set(_default_state_feature_names(self))
        requested_names = default_names | set(self.optional_features)
        unsupported = sorted(requested_names - set(supported_names))
        if unsupported:
            joined = ", ".join(unsupported)
            raise ValueError(f"{self.name} does not support included feature(s): {joined}")
        self.included_features = tuple(name for name in supported_names if name in requested_names)
        self.optional_features = None

    def _allowed_fields(self) -> frozenset[str]:
        match self.name:
            case "course_context":
                return frozenset({"encoding", "included_features", "optional_features"})
            case "control_history":
                return frozenset({"length", "controls", "included_features", "optional_features"})
            case "track_position":
                return frozenset({"progress_source", "included_features", "optional_features"})
            case "vehicle_state" | "machine_context" | "surface_state":
                return frozenset({"included_features", "optional_features"})
            case _:
                raise ValueError(f"Unsupported state component: {self.name!r}")

    def data(self) -> ObservationStateComponentSettings:
        return ObservationStateComponentSettings(
            name=self.name,
            encoding=self.encoding,
            progress_source=self.progress_source,
            length=None if self.length is None else int(self.length),
            controls=self.controls,
            included_features=self.included_features,
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


def _supported_state_feature_names(component: ManagedStateComponentConfig) -> tuple[str, ...]:
    from rl_fzerox.core.envs.observations.state.components import raw_state_component_features

    settings = _component_settings(component, included_features=None)
    return tuple(feature.name for feature in raw_state_component_features(settings))


def _default_state_feature_names(component: ManagedStateComponentConfig) -> tuple[str, ...]:
    from rl_fzerox.core.envs.observations.state.components import state_component_features

    settings = _component_settings(component, included_features=None)
    return tuple(feature.name for feature in state_component_features(settings))


def _component_settings(
    component: ManagedStateComponentConfig,
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
