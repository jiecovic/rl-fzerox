# src/rl_fzerox/core/engine_tuning/state/candidate.py
"""Aggregated per-engine candidate observations."""

from __future__ import annotations

from dataclasses import dataclass, replace


def _clamp_unit_interval(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass(frozen=True, slots=True)
class EngineTuningCandidateState:
    """Aggregated observations for one engine value in one context."""

    context_key: str
    course_key: str
    vehicle_id: str
    engine_setting_raw_value: int
    score_count: int = 0
    episode_count: int = 0
    finish_count: int = 0
    return_count: int = 0
    decayed_count: float = 0.0
    decayed_score_total: float = 0.0
    score_total: float = 0.0
    best_score: float | None = None
    completion_score_total: float = 0.0
    best_completion_score: float | None = None
    finish_score_total: float = 0.0
    best_finish_score: float | None = None
    return_score_total: float = 0.0
    best_return_score: float | None = None
    best_time_ms: int | None = None

    @property
    def active_score_count(self) -> int:
        if self.score_count > 0:
            return self.score_count
        return self.finish_count

    @property
    def observation_count(self) -> int:
        return max(
            self.active_score_count,
            self.episode_count,
            self.finish_count,
            self.return_count,
        )

    @property
    def mean_score(self) -> float | None:
        if self.decayed_count <= 0.0:
            return None
        return self.decayed_score_total / self.decayed_count

    @property
    def raw_mean_score(self) -> float | None:
        if self.active_score_count <= 0:
            return None
        return self.score_total / self.active_score_count

    @property
    def mean_finish_score(self) -> float | None:
        if self.finish_count <= 0:
            return None
        return self.finish_score_total / self.finish_count

    @property
    def mean_completion_score(self) -> float | None:
        if not self.has_valid_episode_statistics:
            return None
        return _clamp_unit_interval(self.completion_score_total / self.episode_count)

    @property
    def finish_rate_score(self) -> float | None:
        if not self.has_valid_episode_statistics:
            return None
        return _clamp_unit_interval(self.finish_count / self.episode_count)

    @property
    def failure_rate_score(self) -> float | None:
        finish_rate = self.finish_rate_score
        return None if finish_rate is None else _clamp_unit_interval(1.0 - finish_rate)

    @property
    def has_valid_episode_statistics(self) -> bool:
        """Return whether episode-derived rates can be interpreted safely."""

        return (
            self.episode_count > 0
            and self.finish_count <= self.episode_count
            and self.completion_score_total <= float(self.episode_count) + 1e-9
        )

    @property
    def mean_return_score(self) -> float | None:
        if self.return_count <= 0:
            return None
        return self.return_score_total / self.return_count

    def record(
        self,
        *,
        score: float | None,
        completion_fraction: float,
        finish_time_ms: int | None,
        episode_return: float | None,
    ) -> EngineTuningCandidateState:
        """Return state after one default-baseline episode observation."""

        next_finish_count = self.finish_count
        next_finish_score_total = self.finish_score_total
        next_best_finish_score = self.best_finish_score
        next_best_time_ms = self.best_time_ms
        if finish_time_ms is not None:
            clamped_finish_time_ms = max(1, int(finish_time_ms))
            finish_score = -float(clamped_finish_time_ms) * 0.001
            next_finish_count += 1
            next_finish_score_total += finish_score
            next_best_finish_score = (
                finish_score
                if next_best_finish_score is None
                else max(next_best_finish_score, finish_score)
            )
            next_best_time_ms = (
                clamped_finish_time_ms
                if next_best_time_ms is None
                else min(next_best_time_ms, clamped_finish_time_ms)
            )
        next_episode_count = self.episode_count
        next_completion = max(0.0, min(1.0, float(completion_fraction)))
        next_completion_score_total = self.completion_score_total + next_completion
        next_best_completion_score = (
            next_completion
            if self.best_completion_score is None
            else max(self.best_completion_score, next_completion)
        )
        next_return_count = self.return_count
        next_return_score_total = self.return_score_total
        next_best_return_score = self.best_return_score
        if episode_return is not None:
            return_score = float(episode_return)
            next_return_count += 1
            next_return_score_total += return_score
            next_best_return_score = (
                return_score
                if next_best_return_score is None
                else max(next_best_return_score, return_score)
            )
        active_score_count = self.score_count
        active_decayed_count = self.decayed_count
        active_decayed_score_total = self.decayed_score_total
        active_score_total = self.score_total
        active_best_score = self.best_score
        if score is not None:
            objective_score = float(score)
            active_score_count += 1
            active_decayed_count += 1.0
            active_decayed_score_total += objective_score
            active_score_total += objective_score
            active_best_score = (
                objective_score
                if active_best_score is None
                else max(active_best_score, objective_score)
            )
        return replace(
            self,
            score_count=active_score_count,
            episode_count=next_episode_count + 1,
            finish_count=next_finish_count,
            return_count=next_return_count,
            decayed_count=active_decayed_count,
            decayed_score_total=active_decayed_score_total,
            score_total=active_score_total,
            best_score=active_best_score,
            completion_score_total=next_completion_score_total,
            best_completion_score=next_best_completion_score,
            finish_score_total=next_finish_score_total,
            best_finish_score=next_best_finish_score,
            return_score_total=next_return_score_total,
            best_return_score=next_best_return_score,
            best_time_ms=next_best_time_ms,
        )

    def decay(self, stat_decay: float) -> EngineTuningCandidateState:
        """Return state with discounted model statistics and intact history fields."""

        clamped_decay = max(0.0, min(0.999999, float(stat_decay)))
        if self.decayed_count <= 0.0 and self.decayed_score_total == 0.0:
            return self
        return replace(
            self,
            decayed_count=self.decayed_count * clamped_decay,
            decayed_score_total=self.decayed_score_total * clamped_decay,
        )
