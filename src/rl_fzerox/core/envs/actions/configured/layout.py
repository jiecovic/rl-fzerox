# src/rl_fzerox/core/envs/actions/configured/layout.py
"""Shared configured-action layout rules.

The configured discrete and hybrid adapters both derive branch sizes, neutral
values, and lean/pitch decoding from the same resolved runtime layout. Keeping
that logic here avoids two subtly diverging interpretations of the same config.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fzerox_emulator import SpinRequest
from fzerox_emulator.arrays import DiscreteAction
from fzerox_emulator.control import spin_request_from_index
from rl_fzerox.core.envs.actions.base import DiscreteActionDimension
from rl_fzerox.core.runtime_spec.schema import ActionRuntimeConfig


@dataclass(frozen=True, slots=True)
class HybridActionLayout:
    """Shape metadata for one configured hybrid action layout."""

    continuous_size: int
    dimensions: tuple[DiscreteActionDimension, ...]

    @property
    def discrete_size(self) -> int:
        return len(self.dimensions)


def configured_dimensions(
    config: ActionRuntimeConfig,
) -> tuple[DiscreteActionDimension, ...]:
    """Return the concrete discrete branches implied by one runtime action config."""

    return tuple(
        DiscreteActionDimension(label=axis, size=discrete_axis_size(axis, config))
        for axis in config.layout_discrete_axes
    )


def discrete_axis_size(axis: str, config: ActionRuntimeConfig) -> int:
    """Return the branch width for one configured discrete axis."""

    if axis == "steer":
        return int(config.steer_buckets)
    if axis in {"gas", "air_brake", "boost"}:
        return 2
    if axis == "lean":
        return 4 if config.lean_output_mode == "four_way_categorical" else 3
    if axis in {"lean_left", "lean_right"}:
        return 2
    if axis == "spin":
        return 3
    if axis == "pitch":
        return int(config.pitch_buckets)
    raise ValueError(f"Unsupported configured discrete axis: {axis!r}")


def idle_discrete_values(
    dimensions: tuple[DiscreteActionDimension, ...],
    config: ActionRuntimeConfig,
) -> DiscreteAction:
    """Return the neutral MultiDiscrete action for one configured layout."""

    values: list[int] = []
    for dimension in dimensions:
        if dimension.label == "steer":
            values.append(config.steer_buckets // 2)
        elif dimension.label == "pitch":
            values.append(config.pitch_buckets // 2)
        else:
            values.append(0)
    return np.asarray(values, dtype=np.int64)


def pitch_bucket_value(index: int, *, bucket_count: int) -> float:
    """Map one discrete pitch bucket back to the emulator stick range."""

    neutral_index = bucket_count // 2
    if neutral_index <= 0:
        return 0.0
    return float(index - neutral_index) / float(neutral_index)


def categorical_lean_state(index: int, *, four_way: bool) -> tuple[bool, bool]:
    """Translate one categorical lean branch value into left/right intent."""

    if index == 1:
        return True, False
    if index == 2:
        return False, True
    if four_way and index == 3:
        return True, True
    return False, False


def spin_request_value(index: int) -> SpinRequest:
    """Translate one 3-way spin branch value into a native macro request."""

    return spin_request_from_index(index)
