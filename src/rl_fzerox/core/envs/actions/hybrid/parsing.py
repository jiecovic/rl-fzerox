# src/rl_fzerox/core/envs/actions/hybrid/parsing.py
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from fzerox_emulator.arrays import ContinuousAction, DiscreteAction
from rl_fzerox.core.domain.hybrid_action import (
    HYBRID_CONTINUOUS_ACTION_KEY,
    HYBRID_DISCRETE_ACTION_KEY,
)
from rl_fzerox.core.envs.actions.base import ActionValue
from rl_fzerox.core.envs.actions.continuous_controls import (
    continuous_action_array,
    discrete_action_array,
    hybrid_branch,
)
from rl_fzerox.core.envs.actions.hybrid.layouts import (
    LEAN_PRIMITIVES,
    PITCH_BUCKETS,
    STEER_DRIVE_AIR_BRAKE_BOOST_LEAN_PITCH_LAYOUT,
    STEER_DRIVE_AIR_BRAKE_PRIMITIVE_LAYOUT,
    STEER_DRIVE_BOOST_LEAN_LAYOUT,
    STEER_DRIVE_LEAN_LAYOUT,
    STEER_GAS_AIR_BRAKE_BOOST_LEAN_LAYOUT,
    STEER_GAS_AIR_BRAKE_BOOST_LEAN_PITCH_LAYOUT,
    STEER_GAS_BOOST_LEAN_LAYOUT,
)


def parse_hybrid_steer_drive_lean(action: ActionValue) -> tuple[float, float, int]:
    continuous_values, discrete_values = parse_hybrid_branches(
        action,
        expected_size=STEER_DRIVE_LEAN_LAYOUT.discrete_size,
        action_label="Hybrid steer-drive-lean discrete",
        field_labels=("lean",),
    )
    lean = int(discrete_values[0])
    if not 0 <= lean < 3:
        raise ValueError(f"Invalid hybrid lean index {lean}")

    steer = float(np.clip(continuous_values[0], -1.0, 1.0))
    drive = float(np.clip(continuous_values[1], -1.0, 1.0))
    return steer, drive, lean


def parse_hybrid_steer_drive_boost_lean_pair(
    action: ActionValue,
) -> tuple[float, float, int, int]:
    continuous_values, discrete_values = parse_hybrid_branches(
        action,
        expected_size=STEER_DRIVE_BOOST_LEAN_LAYOUT.discrete_size,
        action_label="Hybrid steer-drive-boost-lean discrete",
        field_labels=("lean", "boost"),
    )
    lean = int(discrete_values[0])
    boost = int(discrete_values[1])
    if not 0 <= lean < 3:
        raise ValueError(f"Invalid hybrid lean index {lean}")
    if not 0 <= boost < 2:
        raise ValueError(f"Invalid hybrid boost index {boost}")

    steer = float(np.clip(continuous_values[0], -1.0, 1.0))
    drive = float(np.clip(continuous_values[1], -1.0, 1.0))
    return steer, drive, lean, boost


def parse_hybrid_steer_gas_air_brake_boost_lean(
    action: ActionValue,
) -> tuple[float, int, int, int, int]:
    continuous_values, discrete_values = parse_hybrid_branches(
        action,
        continuous_size=STEER_GAS_AIR_BRAKE_BOOST_LEAN_LAYOUT.continuous_size,
        continuous_field_labels=("steer",),
        expected_size=STEER_GAS_AIR_BRAKE_BOOST_LEAN_LAYOUT.discrete_size,
        action_label="Hybrid steer-gas-air-brake-boost-lean discrete",
        field_labels=("gas", "air_brake", "boost", "lean"),
    )
    gas = int(discrete_values[0])
    air_brake = int(discrete_values[1])
    boost = int(discrete_values[2])
    lean = int(discrete_values[3])
    if not 0 <= gas < 2:
        raise ValueError(f"Invalid hybrid gas index {gas}")
    if not 0 <= air_brake < 2:
        raise ValueError(f"Invalid hybrid air-brake index {air_brake}")
    if not 0 <= boost < 2:
        raise ValueError(f"Invalid hybrid boost index {boost}")
    if not 0 <= lean < 3:
        raise ValueError(f"Invalid hybrid lean index {lean}")

    steer = float(np.clip(continuous_values[0], -1.0, 1.0))
    return steer, gas, air_brake, boost, lean


def parse_hybrid_steer_gas_air_brake_boost_lean_pitch(
    action: ActionValue,
) -> tuple[float, int, int, int, int, int]:
    continuous_values, discrete_values = parse_hybrid_branches(
        action,
        continuous_size=STEER_GAS_AIR_BRAKE_BOOST_LEAN_PITCH_LAYOUT.continuous_size,
        continuous_field_labels=("steer",),
        expected_size=STEER_GAS_AIR_BRAKE_BOOST_LEAN_PITCH_LAYOUT.discrete_size,
        action_label="Hybrid steer-gas-air-brake-boost-lean-pitch discrete",
        field_labels=("gas", "air_brake", "boost", "lean", "pitch"),
    )
    gas = int(discrete_values[0])
    air_brake = int(discrete_values[1])
    boost = int(discrete_values[2])
    lean = int(discrete_values[3])
    pitch = int(discrete_values[4])
    if not 0 <= gas < 2:
        raise ValueError(f"Invalid hybrid gas index {gas}")
    if not 0 <= air_brake < 2:
        raise ValueError(f"Invalid hybrid air-brake index {air_brake}")
    if not 0 <= boost < 2:
        raise ValueError(f"Invalid hybrid boost index {boost}")
    if not 0 <= lean < 3:
        raise ValueError(f"Invalid hybrid lean index {lean}")
    if not 0 <= pitch < PITCH_BUCKETS.count:
        raise ValueError(f"Invalid hybrid pitch index {pitch}")

    steer = float(np.clip(continuous_values[0], -1.0, 1.0))
    return steer, gas, air_brake, boost, lean, pitch


def parse_hybrid_steer_drive_air_brake_boost_lean_pitch(
    action: ActionValue,
) -> tuple[float, float, int, int, int, int]:
    continuous_values, discrete_values = parse_hybrid_branches(
        action,
        continuous_size=STEER_DRIVE_AIR_BRAKE_BOOST_LEAN_PITCH_LAYOUT.continuous_size,
        continuous_field_labels=("steer", "drive"),
        expected_size=STEER_DRIVE_AIR_BRAKE_BOOST_LEAN_PITCH_LAYOUT.discrete_size,
        action_label="Hybrid steer-drive-air-brake-boost-lean-pitch discrete",
        field_labels=("air_brake", "boost", "lean", "pitch"),
    )
    air_brake = int(discrete_values[0])
    boost = int(discrete_values[1])
    lean = int(discrete_values[2])
    pitch = int(discrete_values[3])
    if not 0 <= air_brake < 2:
        raise ValueError(f"Invalid hybrid air-brake index {air_brake}")
    if not 0 <= boost < 2:
        raise ValueError(f"Invalid hybrid boost index {boost}")
    if not 0 <= lean < 3:
        raise ValueError(f"Invalid hybrid lean index {lean}")
    if not 0 <= pitch < PITCH_BUCKETS.count:
        raise ValueError(f"Invalid hybrid pitch index {pitch}")

    steer = float(np.clip(continuous_values[0], -1.0, 1.0))
    drive = float(np.clip(continuous_values[1], -1.0, 1.0))
    return steer, drive, air_brake, boost, lean, pitch


def parse_hybrid_steer_gas_boost_lean(
    action: ActionValue,
) -> tuple[float, int, int, int]:
    continuous_values, discrete_values = parse_hybrid_branches(
        action,
        continuous_size=STEER_GAS_BOOST_LEAN_LAYOUT.continuous_size,
        continuous_field_labels=("steer",),
        expected_size=STEER_GAS_BOOST_LEAN_LAYOUT.discrete_size,
        action_label="Hybrid steer-gas-boost-lean discrete",
        field_labels=("gas", "boost", "lean"),
    )
    gas = int(discrete_values[0])
    boost = int(discrete_values[1])
    lean = int(discrete_values[2])
    if not 0 <= gas < 2:
        raise ValueError(f"Invalid hybrid gas index {gas}")
    if not 0 <= boost < 2:
        raise ValueError(f"Invalid hybrid boost index {boost}")
    if not 0 <= lean < 3:
        raise ValueError(f"Invalid hybrid lean index {lean}")

    steer = float(np.clip(continuous_values[0], -1.0, 1.0))
    return steer, gas, boost, lean


def parse_hybrid_steer_drive_boost_lean(
    action: ActionValue,
) -> tuple[float, float, float, int, int]:
    continuous_values, discrete_values = parse_hybrid_branches(
        action,
        continuous_size=STEER_DRIVE_AIR_BRAKE_PRIMITIVE_LAYOUT.continuous_size,
        continuous_field_labels=("steer", "drive", "air_brake"),
        expected_size=STEER_DRIVE_AIR_BRAKE_PRIMITIVE_LAYOUT.discrete_size,
        action_label="Hybrid steer-drive-boost-lean discrete",
        field_labels=("lean", "boost"),
    )
    lean = int(discrete_values[0])
    boost = int(discrete_values[1])
    if not 0 <= lean < LEAN_PRIMITIVES.count:
        raise ValueError(f"Invalid hybrid lean index {lean}")
    if not 0 <= boost < 2:
        raise ValueError(f"Invalid hybrid boost index {boost}")

    steer = float(np.clip(continuous_values[0], -1.0, 1.0))
    drive = float(np.clip(continuous_values[1], -1.0, 1.0))
    air_brake = float(np.clip(continuous_values[2], -1.0, 1.0))
    return steer, drive, air_brake, lean, boost


def with_default_lean_primitive_mask(
    base_overrides: dict[str, tuple[int, ...]] | None,
) -> dict[str, tuple[int, ...]]:
    if base_overrides is not None and "lean" in base_overrides:
        return base_overrides
    overrides = dict(base_overrides or {})
    overrides["lean"] = LEAN_PRIMITIVES.enabled_default
    return overrides


def parse_hybrid_branches(
    action: ActionValue,
    *,
    continuous_size: int = STEER_DRIVE_LEAN_LAYOUT.continuous_size,
    continuous_field_labels: tuple[str, ...] = ("steer", "drive"),
    expected_size: int,
    action_label: str,
    field_labels: tuple[str, ...],
) -> tuple[ContinuousAction, DiscreteAction]:
    if not isinstance(action, Mapping):
        raise ValueError(
            "Hybrid steer-drive actions must be a mapping with 'continuous' and 'discrete' branches"
        )

    continuous_values = continuous_action_array(
        hybrid_branch(action, HYBRID_CONTINUOUS_ACTION_KEY),
        expected_size=continuous_size,
        action_label="Hybrid steer-drive continuous",
        field_labels=continuous_field_labels,
    )
    discrete_values = discrete_action_array(
        hybrid_branch(action, HYBRID_DISCRETE_ACTION_KEY),
        expected_size=expected_size,
        action_label=action_label,
        field_labels=field_labels,
    )
    return continuous_values, discrete_values
