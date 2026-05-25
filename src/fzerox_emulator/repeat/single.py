# src/fzerox_emulator/repeat/single.py
"""Adapter for one native repeated step with a single observation output."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fzerox_emulator.base.observations import ObservationImageRecipe
from fzerox_emulator.base.results import BackendStepResult
from fzerox_emulator.control import RaceControlState
from fzerox_emulator.repeat.requests import native_repeat_observation_request
from fzerox_emulator.repeat.step_options import RepeatStepConfig

if TYPE_CHECKING:
    from fzerox_emulator._native import Emulator as NativeEmulator


def run_repeat_step(
    native: NativeEmulator,
    control_state: RaceControlState,
    *,
    config: RepeatStepConfig,
    recipe: ObservationImageRecipe,
) -> BackendStepResult:
    """Run the native single-observation repeated-step API and wrap the result."""

    observation, summary, status, telemetry = native.step_repeat_raw(
        native_repeat_observation_request(config, control_state, recipe)
    )
    return BackendStepResult(
        observation=observation,
        summary=summary,
        status=status,
        telemetry=telemetry,
    )
