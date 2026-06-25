# src/rl_fzerox/core/policy/auxiliary_state/names.py
"""Literal target-name types for auxiliary-state prediction config.

This module holds the static name vocabulary used by schemas and type checks.
Target metadata, vector slots, and grounded-only support live in ``targets.py``.
"""

from __future__ import annotations

from typing import Literal

type AuxiliaryStateTargetName = Literal[
    "vehicle_state.speed_norm",
    "vehicle_state.energy_frac",
    "vehicle_state.reverse_active",
    "vehicle_state.airborne",
    "vehicle_state.can_boost",
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
