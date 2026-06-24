# src/rl_fzerox/core/domain/observations/components.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from rl_fzerox.core.domain.track_position import height_above_ground_feature

type ActionHistoryControlName = Literal[
    "steer",
    "gas",
    "thrust",
    "air_brake",
    "boost",
    "lean",
    "pitch",
]
type ObservationCourseContextName = Literal["none", "one_hot_builtin"]
type TrackPositionProgressSourceName = Literal[
    "lap_progress",
    "segment_progress",
    "none",
]
type ObservationStateComponentName = Literal[
    "vehicle_state",
    "machine_context",
    "track_position",
    "surface_state",
    "course_context",
    "control_history",
]


@dataclass(frozen=True, slots=True)
class StateComponentFeaturePolicy:
    """Default observation-shape policy for one state component."""

    default_excluded_features: tuple[str, ...] = ()


_FEATURE_POLICY_BY_COMPONENT: dict[
    ObservationStateComponentName,
    StateComponentFeaturePolicy,
] = {
    "vehicle_state": StateComponentFeaturePolicy(),
    "machine_context": StateComponentFeaturePolicy(),
    "track_position": StateComponentFeaturePolicy(
        default_excluded_features=(height_above_ground_feature().name,),
    ),
    "surface_state": StateComponentFeaturePolicy(),
    "course_context": StateComponentFeaturePolicy(),
    "control_history": StateComponentFeaturePolicy(),
}


@dataclass(frozen=True, slots=True)
class ObservationStateComponentSettings:
    """Runtime settings for one scalar-state component."""

    name: ObservationStateComponentName
    encoding: ObservationCourseContextName | None = None
    progress_source: TrackPositionProgressSourceName | None = None
    length: int | None = None
    controls: tuple[ActionHistoryControlName, ...] | None = None
    included_features: tuple[str, ...] | None = None


def default_excluded_state_feature_names(
    component_name: ObservationStateComponentName,
) -> tuple[str, ...]:
    return _FEATURE_POLICY_BY_COMPONENT[component_name].default_excluded_features


def state_feature_default_enabled(
    component_name: ObservationStateComponentName,
    feature_name: str,
) -> bool:
    """Return whether a raw component feature is included by default."""

    return feature_name not in default_excluded_state_feature_names(component_name)


type StateComponentsSettings = tuple[ObservationStateComponentSettings, ...]
