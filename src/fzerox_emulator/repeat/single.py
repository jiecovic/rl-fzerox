# src/fzerox_emulator/repeat/single.py
from __future__ import annotations

from fzerox_emulator._native import Emulator as NativeEmulator
from fzerox_emulator.base import BackendStepResult, ObservationImageRecipe
from fzerox_emulator.control import ControllerState
from fzerox_emulator.repeat.requests import native_repeat_observation_request
from fzerox_emulator.repeat.step_options import RepeatStepConfig


def run_repeat_step(
    native: NativeEmulator,
    controller_state: ControllerState,
    *,
    config: RepeatStepConfig,
    recipe: ObservationImageRecipe,
) -> BackendStepResult:
    observation, summary, status, telemetry = native.step_repeat_raw(
        native_repeat_observation_request(config, controller_state, recipe)
    )
    return BackendStepResult(
        observation=observation,
        summary=summary,
        status=status,
        telemetry=telemetry,
    )
