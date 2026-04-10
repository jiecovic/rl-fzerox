# src/rl_fzerox/core/envs/rewards/common.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from fzerox_emulator import FZeroXTelemetry, StepStatus, StepSummary


@dataclass(frozen=True)
class RewardStep:
    """Reward and debug breakdown for one env step."""

    reward: float
    breakdown: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class RewardSummaryConfig:
    """Native step-summary thresholds required by one reward tracker."""

    energy_loss_epsilon: float


@dataclass(frozen=True)
class RewardActionContext:
    """Action bits the reward function may need for context-dependent shaping."""

    boost_requested: bool = False


class RewardTracker(Protocol):
    """Narrow interface the env engine expects from one reward implementation."""

    def reset(
        self,
        telemetry: FZeroXTelemetry | None,
        *,
        episode_seed: int | None = None,
    ) -> None:
        """Initialize reward state for a new episode."""
        ...

    def summary_config(self) -> RewardSummaryConfig:
        """Return the native aggregation thresholds this reward expects."""
        ...

    def step_summary(
        self,
        summary: StepSummary,
        status: StepStatus,
        telemetry: FZeroXTelemetry | None,
        action_context: RewardActionContext | None = None,
    ) -> RewardStep:
        """Compute one env-step reward from the repeated-step summary."""
        ...

    def info(self, telemetry: FZeroXTelemetry | None) -> dict[str, object]:
        """Expose lightweight reward-state info for logging and watch UI."""
        ...


def apply_event_penalty(
    entered: bool,
    penalty: float,
    label: str,
    breakdown: dict[str, float],
) -> float:
    if not entered:
        return 0.0
    breakdown[label] = penalty
    return penalty


def finish_placement_bonus(*, position: int, total_racers: int, scale: float) -> float:
    clamped_total_racers = max(total_racers, 1)
    clamped_position = min(max(position, 1), clamped_total_racers)
    better_than_last = clamped_total_racers - clamped_position
    return better_than_last * scale
