# src/rl_fzerox/core/engine_tuning/state.py
"""Runtime state for adaptive engine tuning."""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True, slots=True)
class EngineTuningArmState:
    """Observed performance for one engine bin in one tuning context."""

    context_key: str
    course_key: str
    vehicle_id: str
    engine_setting_raw_value: int
    attempts: int = 0
    finished_attempts: int = 0
    decayed_count: float = 0.0
    decayed_score_total: float = 0.0
    completion_total: float = 0.0
    score_total: float = 0.0
    best_score: float | None = None

    @property
    def mean_score(self) -> float | None:
        if self.decayed_count <= 0.0:
            return None
        return self.decayed_score_total / self.decayed_count

    @property
    def raw_mean_score(self) -> float | None:
        if self.attempts <= 0:
            return None
        return self.score_total / self.attempts

    @property
    def finish_rate(self) -> float | None:
        if self.attempts <= 0:
            return None
        return self.finished_attempts / self.attempts

    @property
    def mean_completion(self) -> float | None:
        if self.attempts <= 0:
            return None
        return self.completion_total / self.attempts

    def record(
        self,
        *,
        score: float,
        completion_fraction: float,
        finished: bool,
        stat_decay: float,
    ) -> EngineTuningArmState:
        """Return state after one discounted episode observation."""

        clamped_decay = max(0.0, min(0.999999, float(stat_decay)))
        return replace(
            self,
            attempts=self.attempts + 1,
            finished_attempts=self.finished_attempts + (1 if finished else 0),
            decayed_count=self.decayed_count * clamped_decay + 1.0,
            decayed_score_total=self.decayed_score_total * clamped_decay + float(score),
            completion_total=self.completion_total + max(0.0, min(1.0, completion_fraction)),
            score_total=self.score_total + float(score),
            best_score=(
                float(score) if self.best_score is None else max(self.best_score, float(score))
            ),
        )


@dataclass(frozen=True, slots=True)
class EngineTuningRuntimeState:
    """One persisted adaptive engine-tuning state snapshot."""

    version: int
    update_count: int
    arms: tuple[EngineTuningArmState, ...]

    def arm_map(self) -> dict[tuple[str, int], EngineTuningArmState]:
        """Return arms keyed by context and engine raw value."""

        return {(arm.context_key, arm.engine_setting_raw_value): arm for arm in self.arms}

    def with_arm(self, arm: EngineTuningArmState) -> EngineTuningRuntimeState:
        """Return state with one arm replaced or inserted."""

        replaced = False
        arms: list[EngineTuningArmState] = []
        for existing in self.arms:
            if (
                existing.context_key == arm.context_key
                and existing.engine_setting_raw_value == arm.engine_setting_raw_value
            ):
                arms.append(arm)
                replaced = True
            else:
                arms.append(existing)
        if not replaced:
            arms.append(arm)
        arms.sort(key=lambda item: (item.context_key, item.engine_setting_raw_value))
        return EngineTuningRuntimeState(
            version=self.version,
            update_count=self.update_count + 1,
            arms=tuple(arms),
        )


def empty_engine_tuning_state() -> EngineTuningRuntimeState:
    """Return an empty state snapshot."""

    return EngineTuningRuntimeState(version=1, update_count=0, arms=())
