# src/rl_fzerox/core/envs/limits.py
from __future__ import annotations

from dataclasses import dataclass

from fzerox_emulator import FZeroXTelemetry, StepSummary
from rl_fzerox.core.config.schema import EnvConfig


@dataclass(frozen=True)
class LimitStep:
    """Episode-limit state after one environment step.

    `stalled_steps` is kept as the public field name for compatibility, but it
    now counts consecutive low-speed internal frames rather than no-progress
    frontier frames.
    """

    step_count: int
    stalled_steps: int
    reverse_steps: int
    truncation_reason: str | None = None


class EpisodeLimits:
    """Track timeout, low-speed stuck, and wrong-way truncation per env step.

    The counters still operate in internal-frame units. Rust aggregates the
    repeated inner frames into a `StepSummary`, and this tracker advances the
    carried streaks from that summary once per outer env step.
    """

    def __init__(self, config: EnvConfig) -> None:
        self._max_episode_steps = config.max_episode_steps
        self._stuck_step_limit = config.stuck_step_limit
        self._wrong_way_step_limit = config.wrong_way_step_limit
        self._step_count = 0
        self._low_speed_steps = 0
        self._reverse_steps = 0

    def reset(self, telemetry: FZeroXTelemetry | None) -> None:
        """Reset counters for a fresh episode."""

        self._step_count = 0
        self._low_speed_steps = 0
        self._reverse_steps = 0

    def step_summary(
        self,
        summary: StepSummary,
        telemetry: FZeroXTelemetry | None,
    ) -> LimitStep:
        """Advance counters from one repeated-step summary."""

        self._step_count += summary.frames_run
        truncation_reason = self._timeout_reason()
        if truncation_reason is not None:
            return LimitStep(
                step_count=self._step_count,
                stalled_steps=self._low_speed_steps,
                reverse_steps=self._reverse_steps,
                truncation_reason=truncation_reason,
            )

        if telemetry is None or not telemetry.in_race_mode:
            self._low_speed_steps = 0
            self._reverse_steps = 0
            return LimitStep(
                step_count=self._step_count,
                stalled_steps=self._low_speed_steps,
                reverse_steps=self._reverse_steps,
            )

        if summary.consecutive_low_speed_frames == summary.frames_run:
            self._low_speed_steps += summary.frames_run
        else:
            self._low_speed_steps = summary.consecutive_low_speed_frames
        if summary.consecutive_reverse_frames == summary.frames_run:
            self._reverse_steps += summary.frames_run
        else:
            self._reverse_steps = summary.consecutive_reverse_frames

        return LimitStep(
            step_count=self._step_count,
            stalled_steps=self._low_speed_steps,
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
        if self._low_speed_steps >= self._stuck_step_limit:
            return "stuck"
        return None
