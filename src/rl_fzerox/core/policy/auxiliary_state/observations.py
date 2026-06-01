# src/rl_fzerox/core/policy/auxiliary_state/observations.py
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from fzerox_emulator.arrays import StateVector


def auxiliary_state_targets_field() -> str:
    """Return the dict field name used for hidden auxiliary-state targets."""

    return "auxiliary_state_targets"


def auxiliary_state_targets_from_mapping(
    mapping: Mapping[str, object],
) -> StateVector | None:
    """Return aux targets from one mapping when present as a float array."""

    raw_targets = mapping.get(auxiliary_state_targets_field())
    if not isinstance(raw_targets, np.ndarray):
        return None
    targets: StateVector = np.asarray(raw_targets, dtype=np.float32)
    return targets


def mapping_has_auxiliary_state_targets(mapping: Mapping[str, object]) -> bool:
    """Return whether one mapping already contains the aux-target field."""

    return auxiliary_state_targets_field() in mapping


def mapping_with_auxiliary_state_targets(
    mapping: Mapping[str, object],
    *,
    targets: StateVector,
) -> dict[str, object]:
    """Return one copied mapping with the aux-target field attached."""

    augmented = dict(mapping)
    augmented[auxiliary_state_targets_field()] = targets
    return augmented
