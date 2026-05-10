from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Literal, TypeAlias, TypeGuard

import numpy as np
from gymnasium import spaces

from fzerox_emulator import FZeroXTelemetry
from fzerox_emulator.arrays import Float32Array, StateVector
from rl_fzerox.core.domain.track_position import height_above_ground_feature
from rl_fzerox.core.envs.course_effects import CourseEffect, course_effect_raw, on_refill_surface
from rl_fzerox.core.envs.telemetry import telemetry_boost_active
from rl_fzerox.core.envs.track_bounds import track_edge_state

_BUILTIN_COURSE_COUNT = 24
_SPEED_NORMALIZER_KPH = 1_500.0
_LATERAL_VELOCITY_NORMALIZER = 32.0


def _builtin_course_observation_feature_names() -> tuple[str, ...]:
    return tuple(
        f"course_context.course_builtin_{index:02d}" for index in range(_BUILTIN_COURSE_COUNT)
    )


AuxiliaryStateTargetName: TypeAlias = Literal[
    "vehicle_state.speed_norm",
    "vehicle_state.energy_frac",
    "vehicle_state.reverse_active",
    "vehicle_state.airborne",
    "vehicle_state.boost_unlocked",
    "vehicle_state.boost_active",
    "vehicle_state.lateral_velocity_norm",
    "vehicle_state.sliding_active",
    "track_position.lap_progress",
    "track_position.edge_ratio",
    "track_position.height_above_ground_norm",
    "track_position.outside_track_bounds",
    "surface_state.on_refill_surface",
    "surface_state.on_dirt_surface",
    "surface_state.on_ice_surface",
    "course_context.builtin_course_id",
]
AuxiliaryStateTargetKind: TypeAlias = Literal["scalar", "binary", "categorical"]
AuxiliaryStateDecodedValue: TypeAlias = float | int | dict[str, float | int | None | list[float]]


@dataclass(frozen=True, slots=True)
class AuxiliaryStateTargetSpec:
    name: AuxiliaryStateTargetName
    kind: AuxiliaryStateTargetKind
    vector_start: int
    vector_stop: int


@dataclass(frozen=True, slots=True)
class AuxiliaryStateTargetDefinition:
    name: AuxiliaryStateTargetName
    kind: AuxiliaryStateTargetKind
    low: float
    high: float
    width: int = 1
    supports_grounded_only: bool = False
    observation_feature_names: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class AuxiliaryStateVectorFeature:
    name: str
    low: float
    high: float


@dataclass(frozen=True, slots=True)
class AuxiliaryStateVectorSpec:
    features: tuple[AuxiliaryStateVectorFeature, ...]

    @property
    def count(self) -> int:
        return len(self.features)

    def low_array(self) -> Float32Array:
        return np.array([feature.low for feature in self.features], dtype=np.float32)

    def high_array(self) -> Float32Array:
        return np.array([feature.high for feature in self.features], dtype=np.float32)


_TARGET_DEFINITIONS: tuple[AuxiliaryStateTargetDefinition, ...] = (
    AuxiliaryStateTargetDefinition(
        name="vehicle_state.speed_norm",
        kind="scalar",
        low=0.0,
        high=2.0,
    ),
    AuxiliaryStateTargetDefinition(
        name="vehicle_state.energy_frac",
        kind="scalar",
        low=0.0,
        high=1.0,
    ),
    AuxiliaryStateTargetDefinition(
        name="vehicle_state.reverse_active",
        kind="binary",
        low=0.0,
        high=1.0,
    ),
    AuxiliaryStateTargetDefinition(
        name="vehicle_state.airborne",
        kind="binary",
        low=0.0,
        high=1.0,
    ),
    AuxiliaryStateTargetDefinition(
        name="vehicle_state.boost_unlocked",
        kind="binary",
        low=0.0,
        high=1.0,
    ),
    AuxiliaryStateTargetDefinition(
        name="vehicle_state.boost_active",
        kind="binary",
        low=0.0,
        high=1.0,
    ),
    AuxiliaryStateTargetDefinition(
        name="vehicle_state.lateral_velocity_norm",
        kind="scalar",
        low=-1.0,
        high=1.0,
    ),
    AuxiliaryStateTargetDefinition(
        name="vehicle_state.sliding_active",
        kind="binary",
        low=0.0,
        high=1.0,
    ),
    AuxiliaryStateTargetDefinition(
        name="track_position.lap_progress",
        kind="scalar",
        low=0.0,
        high=1.0,
    ),
    AuxiliaryStateTargetDefinition(
        name="track_position.edge_ratio",
        kind="scalar",
        low=-1.0,
        high=1.0,
        supports_grounded_only=True,
    ),
    AuxiliaryStateTargetDefinition(
        name="track_position.height_above_ground_norm",
        kind="scalar",
        low=0.0,
        high=1.0,
    ),
    AuxiliaryStateTargetDefinition(
        name="track_position.outside_track_bounds",
        kind="binary",
        low=0.0,
        high=1.0,
    ),
    AuxiliaryStateTargetDefinition(
        name="surface_state.on_refill_surface",
        kind="binary",
        low=0.0,
        high=1.0,
    ),
    AuxiliaryStateTargetDefinition(
        name="surface_state.on_dirt_surface",
        kind="binary",
        low=0.0,
        high=1.0,
    ),
    AuxiliaryStateTargetDefinition(
        name="surface_state.on_ice_surface",
        kind="binary",
        low=0.0,
        high=1.0,
    ),
    AuxiliaryStateTargetDefinition(
        name="course_context.builtin_course_id",
        kind="categorical",
        low=0.0,
        high=1.0,
        width=_BUILTIN_COURSE_COUNT,
        observation_feature_names=_builtin_course_observation_feature_names(),
    ),
)

_TARGET_DEFINITIONS_BY_NAME = {definition.name: definition for definition in _TARGET_DEFINITIONS}

_TARGET_DEFINITIONS_BY_OBSERVATION_FEATURE: dict[str, AuxiliaryStateTargetName] = {
    feature_name: definition.name
    for definition in _TARGET_DEFINITIONS
    for feature_name in definition.observation_feature_names
}


def supported_auxiliary_state_target_names() -> tuple[AuxiliaryStateTargetName, ...]:
    return tuple(definition.name for definition in _TARGET_DEFINITIONS)


def is_auxiliary_state_target_name(name: str) -> TypeGuard[AuxiliaryStateTargetName]:
    return name in _TARGET_DEFINITIONS_BY_NAME


def auxiliary_state_target_name_for_feature(
    feature_name: str,
) -> AuxiliaryStateTargetName | None:
    aliased_target_name = _TARGET_DEFINITIONS_BY_OBSERVATION_FEATURE.get(feature_name)
    if aliased_target_name is not None:
        return aliased_target_name
    return feature_name if is_auxiliary_state_target_name(feature_name) else None


def auxiliary_state_target_supports_grounded_only(
    name: AuxiliaryStateTargetName,
) -> bool:
    return _target_definition(name).supports_grounded_only


@lru_cache(maxsize=1)
def _target_specs() -> tuple[AuxiliaryStateTargetSpec, ...]:
    specs: list[AuxiliaryStateTargetSpec] = []
    offset = 0
    for definition in _TARGET_DEFINITIONS:
        width = definition.width
        specs.append(
            AuxiliaryStateTargetSpec(
                name=definition.name,
                kind=definition.kind,
                vector_start=offset,
                vector_stop=offset + width,
            )
        )
        offset += width
    return tuple(specs)


@lru_cache(maxsize=1)
def auxiliary_state_target_spec() -> AuxiliaryStateVectorSpec:
    features: list[AuxiliaryStateVectorFeature] = []
    for spec in _target_specs():
        if spec.kind == "categorical":
            features.extend(
                AuxiliaryStateVectorFeature(
                    name=f"{spec.name}.{index:02d}",
                    low=0.0,
                    high=1.0,
                )
                for index in range(spec.vector_stop - spec.vector_start)
            )
            continue

        definition = _target_definition(spec.name)
        features.append(
            AuxiliaryStateVectorFeature(
                name=spec.name,
                low=definition.low,
                high=definition.high,
            )
        )
    return AuxiliaryStateVectorSpec(features=tuple(features))


def resolve_auxiliary_state_target(name: AuxiliaryStateTargetName) -> AuxiliaryStateTargetSpec:
    for spec in _target_specs():
        if spec.name == name:
            return spec
    raise ValueError(f"Unsupported auxiliary-state target: {name!r}")


def auxiliary_state_target_bounds(name: AuxiliaryStateTargetName) -> tuple[float, float]:
    definition = _target_definition(name)
    return definition.low, definition.high


def _target_definition(name: AuxiliaryStateTargetName) -> AuxiliaryStateTargetDefinition:
    return _TARGET_DEFINITIONS_BY_NAME[name]


def auxiliary_state_target_vector(telemetry: FZeroXTelemetry) -> StateVector:
    vector = np.zeros(auxiliary_state_target_spec().count, dtype=np.float32)
    player = telemetry.player
    energy_frac = (
        0.0 if player.max_energy <= 0.0 else float(player.energy) / float(player.max_energy)
    )
    lateral_velocity = float(player.local_lateral_velocity)
    edge_state = track_edge_state(player)
    edge_ratio = 0.0 if edge_state.ratio is None else _clamp(edge_state.ratio, -1.0, 1.0)
    raw_effect = course_effect_raw(telemetry)

    _set_scalar(
        vector,
        "vehicle_state.speed_norm",
        _clamp(float(player.speed_kph) / _SPEED_NORMALIZER_KPH, 0.0, 2.0),
    )
    _set_scalar(vector, "vehicle_state.energy_frac", _clamp(energy_frac, 0.0, 1.0))
    _set_scalar(vector, "vehicle_state.reverse_active", 1.0 if player.reverse_timer > 0 else 0.0)
    _set_scalar(vector, "vehicle_state.airborne", 1.0 if player.airborne else 0.0)
    _set_scalar(vector, "vehicle_state.boost_unlocked", 1.0 if player.can_boost else 0.0)
    _set_scalar(
        vector,
        "vehicle_state.boost_active",
        1.0 if telemetry_boost_active(telemetry) else 0.0,
    )
    _set_scalar(
        vector,
        "vehicle_state.lateral_velocity_norm",
        _clamp(lateral_velocity / _LATERAL_VELOCITY_NORMALIZER, -1.0, 1.0),
    )
    _set_scalar(
        vector,
        "vehicle_state.sliding_active",
        1.0 if (not player.airborne and abs(lateral_velocity) > 8.0) else 0.0,
    )
    _set_scalar(vector, "track_position.lap_progress", _lap_progress_fraction(telemetry))
    _set_scalar(vector, "track_position.edge_ratio", edge_ratio)
    _set_scalar(
        vector,
        "track_position.height_above_ground_norm",
        height_above_ground_feature().normalize(float(player.height_above_ground)),
    )
    _set_scalar(
        vector,
        "track_position.outside_track_bounds",
        1.0 if edge_state.outside_bounds else 0.0,
    )
    _set_scalar(
        vector,
        "surface_state.on_refill_surface",
        1.0 if on_refill_surface(telemetry) else 0.0,
    )
    _set_scalar(
        vector,
        "surface_state.on_dirt_surface",
        1.0 if raw_effect == CourseEffect.DIRT else 0.0,
    )
    _set_scalar(
        vector,
        "surface_state.on_ice_surface",
        1.0 if raw_effect == CourseEffect.ICE else 0.0,
    )
    _set_course_one_hot(vector, telemetry.course_index)
    return vector


def auxiliary_state_target_vector_or_zeros(
    telemetry: FZeroXTelemetry | None,
) -> StateVector:
    if telemetry is None:
        return np.zeros(auxiliary_state_target_spec().count, dtype=np.float32)
    return auxiliary_state_target_vector(telemetry)


def auxiliary_state_target_values(
    telemetry: FZeroXTelemetry | None,
) -> dict[AuxiliaryStateTargetName, AuxiliaryStateDecodedValue]:
    if telemetry is None:
        return {}

    vector = auxiliary_state_target_vector(telemetry)
    values: dict[AuxiliaryStateTargetName, AuxiliaryStateDecodedValue] = {}
    for spec in _target_specs():
        if spec.kind == "categorical":
            target_slice = vector[spec.vector_start : spec.vector_stop]
            active_indices = np.flatnonzero(target_slice >= 0.5)
            values[spec.name] = {
                "index": int(active_indices[0]) if len(active_indices) == 1 else None
            }
            continue
        values[spec.name] = float(vector[spec.vector_start])
    return values


@lru_cache(maxsize=1)
def auxiliary_state_target_vector_space() -> spaces.Box:
    spec = auxiliary_state_target_spec()
    return spaces.Box(
        low=spec.low_array(),
        high=spec.high_array(),
        shape=(spec.count,),
        dtype=np.float32,
    )


def _set_scalar(
    vector: StateVector,
    name: AuxiliaryStateTargetName,
    value: float,
) -> None:
    spec = resolve_auxiliary_state_target(name)
    vector[spec.vector_start] = float(value)


def _set_course_one_hot(vector: StateVector, course_index: int) -> None:
    spec = resolve_auxiliary_state_target("course_context.builtin_course_id")
    if 0 <= int(course_index) < _BUILTIN_COURSE_COUNT:
        vector[spec.vector_start + int(course_index)] = 1.0


def _lap_progress_fraction(telemetry: FZeroXTelemetry | None) -> float:
    if telemetry is None or telemetry.course_length <= 0.0:
        return 0.0
    return _clamp(float(telemetry.player.lap_distance) / float(telemetry.course_length), 0.0, 1.0)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))
