# src/rl_fzerox/core/envs/rewards/common.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from rl_fzerox.core.emulator.base import StepSummary
    from rl_fzerox.core.game.telemetry import FZeroXTelemetry


@dataclass(frozen=True)
class RewardStep:
    """Reward, terminal state, and debug breakdown for one telemetry sample."""

    reward: float
    terminated: bool
    breakdown: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class RewardSummaryConfig:
    """Native step-summary thresholds required by one reward tracker."""

    reverse_progress_epsilon: float
    energy_loss_epsilon: float


class RewardTracker(Protocol):
    """Narrow interface the env engine expects from one reward implementation."""

    def reset(self, telemetry: FZeroXTelemetry | None) -> None:
        """Initialize reward state for a new episode."""
        ...

    def summary_config(self) -> RewardSummaryConfig:
        """Return the native aggregation thresholds this reward expects."""
        ...

    def step_summary(
        self,
        summary: StepSummary,
        telemetry: FZeroXTelemetry | None,
    ) -> RewardStep:
        """Compute one env-step reward from the repeated-step summary."""
        ...

    def truncation_penalty(self, truncation_reason: str | None) -> tuple[float, str | None]:
        """Return any extra reward penalty that should apply to a truncation."""
        ...


def apply_flag_penalty(
    entered_flags: int,
    flag: int,
    penalty: float,
    label: str,
    breakdown: dict[str, float],
) -> float:
    if not (entered_flags & flag):
        return 0.0
    breakdown[label] = penalty
    return penalty


def finish_placement_bonus(*, position: int, max_race_position: int, scale: float) -> float:
    clamped_position = min(max(position, 1), max_race_position)
    better_than_last = max_race_position - clamped_position
    return better_than_last * scale
