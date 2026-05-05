# src/rl_fzerox/ui/watch/runtime/observation.py
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from rl_fzerox.core.envs.observations import (
    ObservationValue,
    mask_observation_state,
    observation_state,
    state_feature_indices,
)


@dataclass(frozen=True, slots=True)
class _StateComponentTarget:
    name: str
    prefixes: tuple[str, ...]


STATE_COMPONENT_TARGETS = (
    _StateComponentTarget("vehicle_state", ("vehicle_state.",)),
    _StateComponentTarget("machine_context", ("machine_context.",)),
    _StateComponentTarget("track_position", ("track_position.",)),
    _StateComponentTarget("surface_state", ("surface_state.",)),
    _StateComponentTarget("course_context", ("course_context.",)),
    _StateComponentTarget("control_history", ("control_history.", "prev_")),
)


def apply_watch_state_feature_zeroing(
    observation: ObservationValue,
    info: Mapping[str, object],
    *,
    watch_zeroed_features: frozenset[str],
) -> tuple[ObservationValue, dict[str, object]]:
    """Return a watch-local masked observation plus merged zeroing metadata."""

    next_info = dict(info)
    if not watch_zeroed_features:
        next_info["watch_zeroed_state_features"] = ()
        return observation, next_info

    names = _observation_state_feature_names(info)
    state = observation_state(observation)
    if state is None or len(names) != int(state.size):
        next_info["watch_zeroed_state_features"] = tuple(sorted(watch_zeroed_features))
        return observation, _with_union_zeroed_features(
            next_info,
            watch_zeroed_features=watch_zeroed_features,
        )

    zero_targets = _resolved_zero_targets(names, watch_zeroed_features=watch_zeroed_features)
    masked_observation = mask_observation_state(
        observation,
        feature_indices=state_feature_indices(
            names,
            selected_feature_names=zero_targets,
        ),
    )
    return masked_observation, _with_union_zeroed_features(
        next_info,
        watch_zeroed_features=watch_zeroed_features,
    )


def toggle_watch_state_feature(
    zeroed_features: frozenset[str],
    feature_name: str | None,
) -> frozenset[str]:
    if feature_name is None:
        return zeroed_features
    if feature_name in zeroed_features:
        return frozenset(name for name in zeroed_features if name != feature_name)
    return frozenset((*zeroed_features, feature_name))


def _observation_state_feature_names(info: Mapping[str, object]) -> tuple[str, ...]:
    raw_names = info.get("observation_state_features")
    if isinstance(raw_names, tuple) and all(isinstance(name, str) for name in raw_names):
        return raw_names
    if isinstance(raw_names, list) and all(isinstance(name, str) for name in raw_names):
        return tuple(raw_names)
    return ()


def _resolved_zero_targets(
    feature_names: tuple[str, ...],
    *,
    watch_zeroed_features: frozenset[str],
) -> frozenset[str]:
    resolved = set(feature for feature in watch_zeroed_features if feature in feature_names)
    for target in STATE_COMPONENT_TARGETS:
        if target.name not in watch_zeroed_features:
            continue
        resolved.update(
            feature_name
            for feature_name in feature_names
            if any(feature_name.startswith(prefix) for prefix in target.prefixes)
        )
    return frozenset(resolved)


def _with_union_zeroed_features(
    info: dict[str, object],
    *,
    watch_zeroed_features: frozenset[str],
) -> dict[str, object]:
    configured = {
        feature_name
        for feature_name in _zeroed_state_feature_names(info)
        if isinstance(feature_name, str) and feature_name
    }
    info["watch_zeroed_state_features"] = tuple(sorted(watch_zeroed_features))
    info["observation_zeroed_state_features"] = tuple(
        sorted(configured | set(watch_zeroed_features))
    )
    return info


def _zeroed_state_feature_names(info: Mapping[str, object]) -> tuple[str, ...]:
    raw_features = info.get("observation_zeroed_state_features")
    if isinstance(raw_features, tuple) and all(isinstance(name, str) for name in raw_features):
        return raw_features
    if isinstance(raw_features, list) and all(isinstance(name, str) for name in raw_features):
        return tuple(raw_features)
    return ()
