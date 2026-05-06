# src/rl_fzerox/core/envs/observations/masking.py
from __future__ import annotations

from collections.abc import Collection, Sequence

import numpy as np

from fzerox_emulator.arrays import StateVector

from .value import ObservationValue


def state_feature_indices(
    feature_names: Sequence[str],
    *,
    selected_feature_names: Collection[str],
) -> tuple[int, ...]:
    """Return ordered state-vector indices for the selected feature names."""

    selected_names = frozenset(selected_feature_names)
    if not selected_names:
        return ()
    return tuple(
        index for index, feature_name in enumerate(feature_names) if feature_name in selected_names
    )


def mask_observation_state(
    observation: ObservationValue,
    *,
    feature_indices: Sequence[int],
) -> ObservationValue:
    """Return one observation with the selected state-vector indices zeroed."""

    if not feature_indices:
        return observation

    if not isinstance(observation, dict):
        return observation
    state = observation.get("state")
    if not isinstance(state, np.ndarray):
        return observation
    masked_observation = dict(observation)
    masked_observation["state"] = mask_state_vector(state, feature_indices=feature_indices)
    return masked_observation


def mask_state_vector(
    state: StateVector,
    *,
    feature_indices: Sequence[int],
) -> StateVector:
    """Return one copied state vector with the selected indices set to zero."""

    if not feature_indices:
        return state
    masked_state: StateVector = np.array(state, copy=True)
    masked_state[list(feature_indices)] = 0.0
    return masked_state
