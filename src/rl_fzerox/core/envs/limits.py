# src/rl_fzerox/core/envs/limits.py
from __future__ import annotations

from dataclasses import dataclass

from rl_fzerox.core.config.schema import EnvConfig
from rl_fzerox.core.game.telemetry import FZeroXTelemetry


@dataclass(frozen=True)
class LimitStep:
    """Episode-limit state after one environment step."""

    step_count: int
    stalled_steps: int
    reverse_steps: int
    truncation_reason: str | None = None


class EpisodeLimits:
    """Track timeout and no-progress truncation for one rollout."""

    def __init__(self, config: EnvConfig) -> None:
        self._max_episode_steps = config.max_episode_steps
        self._stuck_grace_steps = config.stuck_grace_steps
        self._stuck_step_limit = config.stuck_step_limit
        self._stuck_progress_epsilon = float(config.stuck_progress_epsilon)
        self._wrong_way_step_limit = config.wrong_way_step_limit
        self._wrong_way_progress_epsilon = float(config.wrong_way_progress_epsilon)
        self._step_count = 0
        self._stalled_steps = 0
        self._reverse_steps = 0
        self._best_race_distance = float("-inf")
        self._previous_race_distance = float("-inf")

    def reset(self, telemetry: FZeroXTelemetry | None) -> None:
        """Reset counters for a fresh episode."""

        self._step_count = 0
        self._stalled_steps = 0
        self._reverse_steps = 0
        if telemetry is None:
            self._best_race_distance = float("-inf")
            self._previous_race_distance = float("-inf")
            return
        self._best_race_distance = telemetry.player.race_distance
        self._previous_race_distance = telemetry.player.race_distance

    def step(self, telemetry: FZeroXTelemetry | None) -> LimitStep:
        """Advance counters from the latest telemetry sample."""

        self._step_count += 1
        truncation_reason = self._timeout_reason()
        if truncation_reason is not None:
            return LimitStep(
                step_count=self._step_count,
                stalled_steps=self._stalled_steps,
                reverse_steps=self._reverse_steps,
                truncation_reason=truncation_reason,
            )

        if telemetry is None or not telemetry.in_race_mode:
            self._stalled_steps = 0
            self._reverse_steps = 0
            return LimitStep(
                step_count=self._step_count,
                stalled_steps=self._stalled_steps,
                reverse_steps=self._reverse_steps,
            )

        progress_delta = telemetry.player.race_distance - self._previous_race_distance
        progress_gain = telemetry.player.race_distance - self._best_race_distance
        if progress_gain > self._stuck_progress_epsilon:
            self._best_race_distance = telemetry.player.race_distance
            self._stalled_steps = 0
        else:
            self._stalled_steps += 1
        if progress_delta < -self._wrong_way_progress_epsilon:
            self._reverse_steps += 1
        else:
            self._reverse_steps = 0
        self._previous_race_distance = telemetry.player.race_distance

        return LimitStep(
            step_count=self._step_count,
            stalled_steps=self._stalled_steps,
            reverse_steps=self._reverse_steps,
            truncation_reason=self._truncation_reason(),
        )

    def _timeout_reason(self) -> str | None:
        if self._step_count >= self._max_episode_steps:
            return "timeout"
        return None

    def _truncation_reason(self) -> str | None:
        if self._reverse_steps >= self._wrong_way_step_limit:
            return "wrong_way"
        if self._step_count <= self._stuck_grace_steps:
            return None
        if self._stalled_steps >= self._stuck_step_limit:
            return "stuck"
        return None
