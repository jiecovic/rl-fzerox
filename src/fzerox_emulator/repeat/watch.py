# src/fzerox_emulator/repeat/watch.py
"""Adapter for watch-mode repeated steps that also return display frames."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fzerox_emulator.base.observations import ObservationImageRecipe
from fzerox_emulator.base.results import BackendStepResult
from fzerox_emulator.control import ControllerState
from fzerox_emulator.repeat.requests import native_repeat_observation_request
from fzerox_emulator.repeat.step_options import RepeatStepConfig

if TYPE_CHECKING:
    from fzerox_emulator._native import Emulator as NativeEmulator


def run_repeat_watch_step(
    native: NativeEmulator,
    controller_state: ControllerState,
    *,
    config: RepeatStepConfig,
    recipe: ObservationImageRecipe,
) -> BackendStepResult:
    """Run the native watch repeated-step API and wrap observation/display frames."""

    observation, display_frames, summary, status, telemetry = native.step_repeat_watch_raw(
        native_repeat_observation_request(config, controller_state, recipe)
    )
    return BackendStepResult(
        observation=observation,
        summary=summary,
        status=status,
        telemetry=telemetry,
        display_frames=display_frames,
    )
