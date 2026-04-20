# src/rl_fzerox/apps/recording/models.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from rl_fzerox.core.envs import FZeroXEnv
from rl_fzerox.core.training.inference import PolicyRunner

RecordMode = Literal["stream-all", "probe-then-record"]


@dataclass(frozen=True)
class RecordAttemptResult:
    """Outcome from one recorded policy episode."""

    attempt: int
    path: Path
    matched: bool
    finish_rank: int | None
    episode_return: float
    episode_steps: int
    race_time_ms: int
    termination_reason: str | None
    truncation_reason: str | None


@dataclass(frozen=True)
class AttemptRunResult:
    """Internal attempt result from one temp-recorded episode."""

    attempt: int
    path: Path
    matched: bool
    finish_rank: int | None
    episode_return: float
    episode_steps: int
    race_time_ms: int
    termination_reason: str | None
    truncation_reason: str | None


@dataclass(frozen=True)
class RecordingSession:
    """Open emulator env plus policy runner for one recording/probing pass."""

    env: FZeroXEnv
    policy_runner: PolicyRunner
    output_fps: float
