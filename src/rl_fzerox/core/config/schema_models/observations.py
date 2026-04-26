# src/rl_fzerox/core/config/schema_models/observations.py
from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveInt,
    field_validator,
    model_validator,
)

from rl_fzerox.core.config.schema_models.common import (
    ObservationPresetName,
    ObservationResizeFilter,
)
from rl_fzerox.core.domain.observation_components import (
    ActionHistoryControlName,
    ObservationCourseContextName,
    ObservationGroundEffectContextName,
    ObservationStateComponentName,
    ObservationStateComponentSettings,
    ObservationStateProfileName,
)


class ObservationStateComponentConfig(BaseModel):
    """One ordered scalar-state component in the image-state observation."""

    model_config = ConfigDict(extra="forbid")

    name: ObservationStateComponentName
    encoding: ObservationCourseContextName | None = None
    state_profile: ObservationStateProfileName | None = None
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
                "state_profile",
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
            case "vehicle_state" | "machine_context" | "track_position" | "surface_state":
                return frozenset()
            case _:
                raise ValueError(f"Unsupported state component: {self.name!r}")

    def data(self) -> ObservationStateComponentSettings:
        """Return the compact ordered form consumed by env code."""

        return ObservationStateComponentSettings(
            name=self.name,
            encoding=self.encoding,
            state_profile=self.state_profile,
            length=None if self.length is None else int(self.length),
            controls=self.controls,
        )


class ObservationConfig(BaseModel):
    """Observation adapter settings for the current env."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["image", "image_state"] = "image"
    state_profile: Literal[
        "default",
        "steer_history",
        "race_core",
    ] = "default"
    preset: ObservationPresetName = "crop_116x164"
    frame_stack: PositiveInt = 4
    stack_mode: Literal["rgb", "gray", "luma_chroma"] = "rgb"
    minimap_layer: bool = False
    resize_filter: ObservationResizeFilter = "nearest"
    minimap_resize_filter: ObservationResizeFilter = "nearest"
    course_context: ObservationCourseContextName = "none"
    ground_effect_context: ObservationGroundEffectContextName = "none"
    action_history_len: PositiveInt | None = Field(default=None, le=16)
    action_history_controls: tuple[ActionHistoryControlName, ...] = (
        "steer",
        "gas",
        "boost",
        "lean",
    )
    state_components: tuple[ObservationStateComponentConfig, ...] | None = None
    zeroed_state_components: tuple[ObservationStateComponentName, ...] = ()
    zeroed_state_features: tuple[str, ...] = ()

    @field_validator("action_history_controls")
    @classmethod
    def _validate_unique_action_history_controls(
        cls,
        value: tuple[ActionHistoryControlName, ...],
    ) -> tuple[ActionHistoryControlName, ...]:
        if len(set(value)) != len(value):
            raise ValueError("action_history_controls must not contain duplicates")
        normalized = {"gas" if control == "thrust" else control for control in value}
        if len(normalized) != len(value):
            raise ValueError("action_history_controls cannot contain both gas and thrust")
        return value

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

    @field_validator("zeroed_state_components")
    @classmethod
    def _validate_unique_zeroed_state_components(
        cls,
        value: tuple[ObservationStateComponentName, ...],
    ) -> tuple[ObservationStateComponentName, ...]:
        if len(set(value)) != len(value):
            raise ValueError("observation.zeroed_state_components must not contain duplicates")
        return value

    @field_validator("zeroed_state_features")
    @classmethod
    def _validate_unique_zeroed_state_features(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            raise ValueError("observation.zeroed_state_features must not contain duplicates")
        return value

    @model_validator(mode="after")
    def _validate_zeroed_state_components_are_active(self) -> ObservationConfig:
        if not self.zeroed_state_components:
            return self
        if self.state_components is None:
            raise ValueError(
                "observation.zeroed_state_components requires observation.state_components"
            )
        active_names = {str(component.name) for component in self.state_components}
        unknown_names = sorted(
            str(component_name)
            for component_name in self.zeroed_state_components
            if str(component_name) not in active_names
        )
        if unknown_names:
            joined = ", ".join(unknown_names)
            raise ValueError(
                "observation.zeroed_state_components must reference active state components: "
                f"{joined}"
            )
        return self

    def state_components_data(self) -> tuple[ObservationStateComponentSettings, ...] | None:
        """Return state-component settings in the plain form consumed by env code."""

        if self.state_components is None:
            return None
        return tuple(component.data() for component in self.state_components)
