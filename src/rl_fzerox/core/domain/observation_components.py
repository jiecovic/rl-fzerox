# src/rl_fzerox/core/domain/observation_components.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

ActionHistoryControlName: TypeAlias = Literal[
    "steer",
    "gas",
    "thrust",
    "air_brake",
    "boost",
    "lean",
    "pitch",
]
ObservationCourseContextName: TypeAlias = Literal["none", "one_hot_builtin"]
ObservationGroundEffectContextName: TypeAlias = Literal["none", "effect_flags"]
ObservationStateProfileName: TypeAlias = Literal[
    "default",
    "steer_history",
    "race_core",
]
ObservationStateComponentName: TypeAlias = Literal[
    "vehicle_state",
    "machine_context",
    "track_position",
    "surface_state",
    "course_context",
    "legacy_state",
    "control_history",
]


@dataclass(frozen=True, slots=True)
class ObservationStateComponentSettings:
    """Runtime settings for one scalar-state component."""

    name: ObservationStateComponentName
    encoding: ObservationCourseContextName | None = None
    state_profile: ObservationStateProfileName | None = None
    length: int | None = None
    controls: tuple[ActionHistoryControlName, ...] | None = None


StateComponentsSettings: TypeAlias = tuple[ObservationStateComponentSettings, ...]
