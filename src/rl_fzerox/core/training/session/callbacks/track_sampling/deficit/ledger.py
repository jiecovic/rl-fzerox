# src/rl_fzerox/core/training/session/callbacks/track_sampling/deficit/ledger.py
from __future__ import annotations

from collections.abc import Mapping
from math import ceil
from random import Random

from rl_fzerox.core.envs.engine.reset.track_sampling import TrackSamplingDeficitLane
from rl_fzerox.core.training.session.callbacks.track_sampling.state import (
    DeficitBudgetCourseSchedulerState,
    DeficitBudgetSchedulerState,
)

DEFICIT_LANES: tuple[TrackSamplingDeficitLane, ...] = ("uniform", "adaptive")


class DeficitBudgetLedger:
    """Own scheduler debt, reservations, and uniform-lane staleness."""

    def __init__(
        self,
        *,
        course_keys: tuple[str, ...],
        seed: int,
    ) -> None:
        self._course_keys = course_keys
        self._rng = Random(seed)
        self._deficit_steps = {
            lane: {course_key: 0.0 for course_key in course_keys} for lane in DEFICIT_LANES
        }
        self._reserved_reset_steps = {
            lane: {course_key: 0.0 for course_key in course_keys} for lane in DEFICIT_LANES
        }
        self._lane_deficit_steps = {lane: 0.0 for lane in DEFICIT_LANES}
        self._lane_reserved_reset_steps = {lane: 0.0 for lane in DEFICIT_LANES}
        self._scheduler_env_steps = {course_key: 0 for course_key in course_keys}
        self._uniform_assignment_count = 0
        self._last_uniform_assignment_index = {course_key: 0 for course_key in course_keys}

    def add_rollout_budget(
        self,
        *,
        steps: int,
        uniform_fraction: float,
        adaptive_fractions: Mapping[str, float],
    ) -> None:
        uniform_fraction = _clamped_fraction(uniform_fraction)
        adaptive_fraction = 1.0 - uniform_fraction
        uniform_share = 1.0 / len(self._course_keys)
        self._lane_deficit_steps["uniform"] += steps * uniform_fraction
        self._lane_deficit_steps["adaptive"] += steps * adaptive_fraction
        self.clear_reserved_assignments()
        for course_key in self._course_keys:
            self._deficit_steps["uniform"][course_key] += steps * uniform_fraction * uniform_share
            self._deficit_steps["adaptive"][course_key] += (
                steps * adaptive_fraction * adaptive_fractions[course_key]
            )

    def clear_reserved_assignments(self) -> None:
        for lane in DEFICIT_LANES:
            self._lane_reserved_reset_steps[lane] = 0.0
            for course_key in self._course_keys:
                self._reserved_reset_steps[lane][course_key] = 0.0

    def record_scheduler_step(self, course_key: str) -> None:
        self._scheduler_env_steps[course_key] += 1

    def set_scheduler_env_steps(self, course_key: str, value: int) -> None:
        self._scheduler_env_steps[course_key] = max(0, int(value))

    def ensure_scheduler_env_steps_at_least(self, course_key: str, value: int) -> None:
        self._scheduler_env_steps[course_key] = max(
            self._scheduler_env_steps[course_key],
            max(0, int(value)),
        )

    def record_deficit_step(
        self,
        *,
        course_key: str,
        lane: TrackSamplingDeficitLane | None,
        uniform_fraction: float,
    ) -> None:
        if lane is not None:
            self._lane_deficit_steps[lane] -= 1.0
            self._deficit_steps[lane][course_key] -= 1.0
            self._consume_reserved_step(lane, course_key)
            return
        uniform_fraction = _clamped_fraction(uniform_fraction)
        self._lane_deficit_steps["uniform"] -= uniform_fraction
        self._lane_deficit_steps["adaptive"] -= 1.0 - uniform_fraction
        self._deficit_steps["uniform"][course_key] -= uniform_fraction
        self._deficit_steps["adaptive"][course_key] -= 1.0 - uniform_fraction

    def next_lane(self, *, uniform_fraction: float) -> TrackSamplingDeficitLane:
        uniform_fraction = _clamped_fraction(uniform_fraction)
        if uniform_fraction >= 1.0:
            return "uniform"
        if uniform_fraction <= 0.0:
            return "adaptive"
        return max(
            DEFICIT_LANES,
            key=lambda lane: (
                self._lane_deficit_steps[lane] - self._lane_reserved_reset_steps[lane],
                1.0 if lane == "uniform" else 0.0,
            ),
        )

    def next_course_key(
        self,
        *,
        lane: TrackSamplingDeficitLane,
        uniform_staleness_rotations: float,
    ) -> str:
        stale_course_key = (
            self._stale_uniform_course_key(uniform_staleness_rotations)
            if lane == "uniform"
            else None
        )
        if stale_course_key is not None:
            return stale_course_key
        return max(
            self._course_keys,
            key=lambda course_key: (
                self._deficit_steps[lane][course_key]
                - self._reserved_reset_steps[lane][course_key],
                self._rng.random() * 1e-9,
            ),
        )

    def reserve_course_assignment(
        self,
        *,
        lane: TrackSamplingDeficitLane,
        course_key: str,
        assignment_cost: float,
    ) -> None:
        cost = max(1.0, float(assignment_cost))
        self._reserved_reset_steps[lane][course_key] += cost
        self._lane_reserved_reset_steps[lane] += cost
        if lane == "uniform":
            self._uniform_assignment_count += 1
            self._last_uniform_assignment_index[course_key] = self._uniform_assignment_count

    def lane_deficit_steps(self, lane: TrackSamplingDeficitLane) -> float:
        return self._lane_deficit_steps[lane]

    def uniform_stale_course_count(self, *, uniform_staleness_rotations: float) -> int:
        max_gap = self._uniform_staleness_max_assignment_gap(uniform_staleness_rotations)
        if max_gap <= 0:
            return 0
        return sum(
            1
            for course_key in self._course_keys
            if self._uniform_assignment_count - self._last_uniform_assignment_index[course_key]
            >= max_gap
        )

    def uniform_staleness_max_assignment_gap(
        self,
        *,
        uniform_staleness_rotations: float,
    ) -> int:
        return self._uniform_staleness_max_assignment_gap(uniform_staleness_rotations)

    def restore_scheduler_state(self, state: DeficitBudgetSchedulerState | None) -> bool:
        if state is None:
            return False
        restored_entries = {entry.course_key: entry for entry in state.entries}
        if not set(self._course_keys).issubset(restored_entries):
            return False
        restored_any = False
        for course_key in self._course_keys:
            entry = restored_entries.get(course_key)
            if entry is None:
                continue
            restored_any = True
            self._deficit_steps["uniform"][course_key] = float(entry.uniform_deficit_steps)
            self._deficit_steps["adaptive"][course_key] = float(entry.adaptive_deficit_steps)
            self._scheduler_env_steps[course_key] = max(0, int(entry.scheduler_env_steps))
            self._last_uniform_assignment_index[course_key] = max(
                0,
                int(entry.last_uniform_assignment_index),
            )
        if not restored_any:
            return False
        self._lane_deficit_steps["uniform"] = float(state.uniform_lane_deficit_steps)
        self._lane_deficit_steps["adaptive"] = float(state.adaptive_lane_deficit_steps)
        self._uniform_assignment_count = max(0, int(state.uniform_assignment_count))
        return True

    def seed_legacy_deficit_steps_from_accounted_steps(
        self,
        *,
        accounted_env_steps: Mapping[str, int],
        uniform_fraction: float,
        adaptive_fractions: Mapping[str, float],
    ) -> None:
        total_steps = sum(accounted_env_steps.values())
        if total_steps <= 0:
            return
        uniform_fraction = _clamped_fraction(uniform_fraction)
        adaptive_fraction = 1.0 - uniform_fraction
        uniform_share = 1.0 / len(self._course_keys)
        self._lane_deficit_steps["uniform"] = 0.0
        self._lane_deficit_steps["adaptive"] = 0.0
        for course_key in self._course_keys:
            actual_steps = float(accounted_env_steps[course_key])
            self._deficit_steps["uniform"][course_key] = uniform_fraction * (
                total_steps * uniform_share - actual_steps
            )
            self._deficit_steps["adaptive"][course_key] = adaptive_fraction * (
                total_steps * adaptive_fractions[course_key] - actual_steps
            )
            self._lane_deficit_steps["uniform"] += self._deficit_steps["uniform"][course_key]
            self._lane_deficit_steps["adaptive"] += self._deficit_steps["adaptive"][course_key]
            self.ensure_scheduler_env_steps_at_least(course_key, accounted_env_steps[course_key])

    def state(self) -> DeficitBudgetSchedulerState:
        return DeficitBudgetSchedulerState(
            uniform_lane_deficit_steps=self._lane_deficit_steps["uniform"],
            adaptive_lane_deficit_steps=self._lane_deficit_steps["adaptive"],
            uniform_assignment_count=self._uniform_assignment_count,
            entries=tuple(
                DeficitBudgetCourseSchedulerState(
                    course_key=course_key,
                    uniform_deficit_steps=self._deficit_steps["uniform"][course_key],
                    adaptive_deficit_steps=self._deficit_steps["adaptive"][course_key],
                    scheduler_env_steps=self._scheduler_env_steps[course_key],
                    last_uniform_assignment_index=self._last_uniform_assignment_index[course_key],
                )
                for course_key in self._course_keys
            ),
        )

    def _consume_reserved_step(
        self,
        lane: TrackSamplingDeficitLane,
        course_key: str,
    ) -> None:
        course_reserved = self._reserved_reset_steps[lane][course_key]
        if course_reserved > 0.0:
            self._reserved_reset_steps[lane][course_key] = max(0.0, course_reserved - 1.0)
        lane_reserved = self._lane_reserved_reset_steps[lane]
        if lane_reserved > 0.0:
            self._lane_reserved_reset_steps[lane] = max(0.0, lane_reserved - 1.0)

    def _stale_uniform_course_key(self, uniform_staleness_rotations: float) -> str | None:
        max_gap = self._uniform_staleness_max_assignment_gap(uniform_staleness_rotations)
        if max_gap <= 0:
            return None
        candidates = tuple(
            course_key
            for course_key in self._course_keys
            if self._uniform_assignment_count - self._last_uniform_assignment_index[course_key]
            >= max_gap
        )
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda course_key: (
                self._uniform_assignment_count - self._last_uniform_assignment_index[course_key],
                self._deficit_steps["uniform"][course_key]
                - self._reserved_reset_steps["uniform"][course_key],
                self._rng.random() * 1e-9,
            ),
        )

    def _uniform_staleness_max_assignment_gap(
        self,
        uniform_staleness_rotations: float,
    ) -> int:
        rotations = max(0.0, float(uniform_staleness_rotations))
        if rotations <= 0.0:
            return 0
        return max(len(self._course_keys), ceil(len(self._course_keys) * rotations))


def _clamped_fraction(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
