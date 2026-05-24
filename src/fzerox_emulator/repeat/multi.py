# src/fzerox_emulator/repeat/multi.py
from __future__ import annotations

from collections.abc import Sequence

from fzerox_emulator._native import Emulator as NativeEmulator
from fzerox_emulator.base import BackendMultiObservationStepResult, ObservationImageRecipe
from fzerox_emulator.control import ControllerState
from fzerox_emulator.repeat.requests import native_repeat_multi_observation_request
from fzerox_emulator.repeat.step_options import RepeatStepConfig


def run_repeat_multi_observation_step(
    native: NativeEmulator,
    controller_state: ControllerState,
    *,
    config: RepeatStepConfig,
    recipes: Sequence[ObservationImageRecipe],
) -> BackendMultiObservationStepResult:
    if len(recipes) == 0:
        raise ValueError("At least one observation recipe is required")

    observations, summary, status, telemetry = native.step_repeat_multi_observation_raw(
        native_repeat_multi_observation_request(config, controller_state, recipes)
    )
    return BackendMultiObservationStepResult(
        observations=tuple(observations),
        summary=summary,
        status=status,
        telemetry=telemetry,
    )
