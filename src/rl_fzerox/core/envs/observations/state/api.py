# src/rl_fzerox/core/envs/observations/state/api.py
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from fzerox_emulator import FZeroXTelemetry
from fzerox_emulator.arrays import StateVector
from rl_fzerox.core.domain.observation_components import StateComponentsSettings

from .components import (
    action_history_settings_for_observation,
    component_state_values,
    state_vector_spec_from_components,
)
from .history import action_history_feature_names
from .types import StateVectorSpec


def telemetry_state_vector(
    telemetry: FZeroXTelemetry | None,
    *,
    state_components: StateComponentsSettings,
    action_history: Mapping[str, float] | None = None,
    split_lean_history: bool = False,
) -> StateVector:
    """Build the normalized scalar policy-state vector from live game telemetry."""

    values = component_state_values(
        telemetry,
        state_components=state_components,
        action_history=action_history or {},
        split_lean_history=split_lean_history,
    )
    expected_count = state_vector_spec(
        state_components=state_components,
        split_lean_history=split_lean_history,
    ).count
    if len(values) != expected_count:
        raise ValueError(
            "Observation state component value count does not match feature spec: "
            f"{len(values)} != {expected_count}"
        )
    return np.array(values, dtype=np.float32)


def state_vector_spec(
    *,
    state_components: StateComponentsSettings,
    split_lean_history: bool = False,
) -> StateVectorSpec:
    """Return the scalar-state schema selected by config."""

    return state_vector_spec_from_components(
        state_components,
        split_lean_history=split_lean_history,
    )


def state_feature_names(
    *,
    state_components: StateComponentsSettings,
    split_lean_history: bool = False,
) -> tuple[str, ...]:
    """Return ordered scalar-state feature names for the selected components."""

    return state_vector_spec(
        state_components=state_components,
        split_lean_history=split_lean_history,
    ).names


def state_feature_count(
    *,
    state_components: StateComponentsSettings,
    split_lean_history: bool = False,
) -> int:
    """Return scalar-state width for the selected components."""

    return state_vector_spec(
        state_components=state_components,
        split_lean_history=split_lean_history,
    ).count


__all__ = [
    "action_history_feature_names",
    "action_history_settings_for_observation",
    "state_feature_count",
    "state_feature_names",
    "state_vector_spec",
    "telemetry_state_vector",
]
