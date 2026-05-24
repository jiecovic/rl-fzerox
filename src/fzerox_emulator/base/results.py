# src/fzerox_emulator/base/results.py
from __future__ import annotations

from dataclasses import dataclass, field

from fzerox_emulator._native import FZeroXTelemetry, StepStatus, StepSummary
from fzerox_emulator.arrays import ObservationFrame, RgbFrame


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
    display_frames: tuple[RgbFrame, ...] = ()


@dataclass(frozen=True)
class BackendMultiObservationStepResult:
    """Native repeated-step payload with multiple rendered observation views."""

    observations: tuple[ObservationFrame, ...]
    summary: StepSummary
    status: StepStatus
    telemetry: FZeroXTelemetry | None
