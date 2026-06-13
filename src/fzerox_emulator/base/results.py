# src/fzerox_emulator/base/results.py
"""Small result value objects returned by emulator backend operations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from fzerox_emulator.arrays import (
    AudioFrameCounts,
    ControllerMaskBatch,
    DisplayFrames,
    ObservationFrame,
    Pcm16Samples,
    RgbFrame,
)

if TYPE_CHECKING:
    from fzerox_emulator._native import FZeroXTelemetry, StepStatus, StepSummary


@dataclass(frozen=True)
class ResetState:
    """State returned immediately after an emulator reset."""

    frame: RgbFrame
    info: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class FrameStep:
    """One emulator frame worth of output and metadata."""

    frame: RgbFrame
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class BackendStepResult:
    """Native repeated-step payload consumed by the env engine."""

    observation: ObservationFrame
    summary: StepSummary
    status: StepStatus
    telemetry: FZeroXTelemetry | None
    display_frames: DisplayFrames = ()
    display_controller_masks: ControllerMaskBatch = ()
    audio_samples: Pcm16Samples = ()
    audio_frame_counts: AudioFrameCounts = ()


@dataclass(frozen=True)
class BackendMultiObservationStepResult:
    """Native repeated-step payload with multiple rendered observation views."""

    observations: tuple[ObservationFrame, ...]
    summary: StepSummary
    status: StepStatus
    telemetry: FZeroXTelemetry | None
