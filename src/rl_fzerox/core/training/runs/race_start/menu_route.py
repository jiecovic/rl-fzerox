# src/rl_fzerox/core/training/runs/race_start/menu_route.py
"""Static menu route definitions used by race-start materialization."""

from __future__ import annotations

from dataclasses import dataclass

from rl_fzerox.core.runtime_spec.vehicle_catalog import vehicle_menu_row_and_column
from rl_fzerox.core.training.runs.race_start.models import RaceStartVariant


@dataclass(frozen=True, slots=True)
class MachineSelectRoute:
    """Physical menu route from the default machine slot to one target machine."""

    down_count: int
    right_count: int


def machine_select_route_for_slot(machine_select_slot: int) -> MachineSelectRoute:
    """Return the machine-select menu route for a stock machine slot."""

    row, column = vehicle_menu_row_and_column(machine_select_slot)
    return MachineSelectRoute(down_count=row, right_count=column)


def machine_select_route_for_variant(variant: RaceStartVariant) -> MachineSelectRoute:
    """Return the materializer machine-select route for one race-start variant."""

    slot = (
        variant.machine_select_slot
        if variant.machine_select_slot is not None
        else variant.character_index
    )
    return machine_select_route_for_slot(slot)
