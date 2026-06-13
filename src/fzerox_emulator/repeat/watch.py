# src/fzerox_emulator/repeat/watch.py
"""Adapter for watch-mode repeated steps that also return display frames."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fzerox_emulator.base.observations import ObservationImageRecipe
from fzerox_emulator.base.results import BackendStepResult
from fzerox_emulator.control import RaceControlState
from fzerox_emulator.repeat.requests import native_repeat_observation_request
from fzerox_emulator.repeat.step_options import RepeatStepConfig

if TYPE_CHECKING:
    from fzerox_emulator._native import Emulator as NativeEmulator


def run_repeat_watch_step(
    native: NativeEmulator,
    control_state: RaceControlState,
    *,
    config: RepeatStepConfig,
    recipe: ObservationImageRecipe,
    capture_audio: bool = False,
) -> BackendStepResult:
    """Run the native watch repeated-step API and wrap observation/display frames."""

    (
        observation,
        display_frames,
        display_controller_masks,
        audio_samples,
        audio_frame_counts,
        summary,
        status,
        telemetry,
    ) = native.step_repeat_watch_raw(
        native_repeat_observation_request(
            config,
            control_state,
            recipe,
            capture_audio=capture_audio,
        )
    )
    return BackendStepResult(
        observation=observation,
        summary=summary,
        status=status,
        telemetry=telemetry,
        display_frames=display_frames,
        display_controller_masks=display_controller_masks,
        audio_samples=audio_samples,
        audio_frame_counts=audio_frame_counts,
    )
