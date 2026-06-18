# src/rl_fzerox/core/engine_tuning/state.py
"""Runtime state for adaptive engine tuning."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from rl_fzerox.core.engine_tuning.types import EngineTunerBackend, EngineTunerObjective

ENGINE_TUNING_STATE_VERSION = 7


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
        return max(self.active_score_count, self.episode_count, self.finish_count)

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
        if self.episode_count <= 0:
            return None
        return self.completion_score_total / self.episode_count

    @property
    def finish_rate_score(self) -> float | None:
        if self.episode_count <= 0:
            return None
        return self.finish_count / self.episode_count

    @property
    def failure_rate_score(self) -> float | None:
        finish_rate = self.finish_rate_score
        return None if finish_rate is None else 1.0 - finish_rate

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

    def with_active_objective(
        self,
        objective: EngineTunerObjective,
    ) -> EngineTuningCandidateState | None:
        """Return this candidate with active scoring rebuilt for an objective."""

        if objective == "episode_return":
            if self.return_count <= 0:
                return None
            return replace(
                self,
                score_count=self.return_count,
                decayed_count=float(self.return_count),
                decayed_score_total=self.return_score_total,
                score_total=self.return_score_total,
                best_score=self.best_return_score,
            )
        if objective == "completion":
            if self.episode_count <= 0:
                return None
            return replace(
                self,
                score_count=self.episode_count,
                decayed_count=float(self.episode_count),
                decayed_score_total=self.completion_score_total,
                score_total=self.completion_score_total,
                best_score=self.best_completion_score,
            )
        if objective == "finish_rate":
            if self.episode_count <= 0:
                return None
            best_score = 1.0 if self.finish_count > 0 else 0.0
            finish_score_total = float(self.finish_count)
            return replace(
                self,
                score_count=self.episode_count,
                decayed_count=float(self.episode_count),
                decayed_score_total=finish_score_total,
                score_total=finish_score_total,
                best_score=best_score,
            )
        if self.finish_count <= 0:
            return None
        finish_score_total = (
            self.finish_score_total
            if self.finish_score_total != 0.0 or self.best_finish_score is not None
            else self.score_total
        )
        best_finish_score = (
            self.best_finish_score if self.best_finish_score is not None else self.best_score
        )
        return replace(
            self,
            score_count=self.finish_count,
            decayed_count=float(self.finish_count),
            decayed_score_total=finish_score_total,
            score_total=finish_score_total,
            best_score=best_finish_score,
        )

    def without_return_observations(self) -> EngineTuningCandidateState:
        """Return this candidate with reward-dependent return aggregates removed."""

        return replace(
            self,
            return_count=0,
            return_score_total=0.0,
            best_return_score=None,
        )


@dataclass(frozen=True, slots=True)
class EngineTuningTensorState:
    """One tensor stored in the engine-tuner model checkpoint."""

    name: str
    value: torch.Tensor


@dataclass(frozen=True, slots=True)
class EngineTuningEnsembleMemberState:
    """One persisted MLP ensemble member."""

    tensors: tuple[EngineTuningTensorState, ...]


@dataclass(frozen=True, slots=True)
class EngineTuningModelContextState:
    """Observed context metadata for model-backed tuners."""

    context_key: str
    course_key: str
    vehicle_id: str
    finish_count: int = 0


@dataclass(frozen=True, slots=True)
class EngineTuningModelState:
    """Optional learned model state for non-aggregate tuner backends."""

    backend: EngineTunerBackend
    course_keys: tuple[str, ...]
    vehicle_ids: tuple[str, ...]
    members: tuple[EngineTuningEnsembleMemberState, ...] = ()
    contexts: tuple[EngineTuningModelContextState, ...] = ()


@dataclass(frozen=True, slots=True)
class EngineTuningRuntimeState:
    """One persisted adaptive engine-tuning state snapshot."""

    version: int
    update_count: int
    candidates: tuple[EngineTuningCandidateState, ...]
    objective: EngineTunerObjective = "finish_time"
    reward_fingerprint: str | None = None
    model_state: EngineTuningModelState | None = None

    def candidate_map(self) -> dict[tuple[str, int], EngineTuningCandidateState]:
        """Return candidates keyed by context and engine raw value."""

        return {
            (candidate.context_key, candidate.engine_setting_raw_value): candidate
            for candidate in self.candidates
        }

    def with_candidate(self, candidate: EngineTuningCandidateState) -> EngineTuningRuntimeState:
        """Return state with one candidate observation aggregate replaced or inserted."""

        replaced = False
        candidates: list[EngineTuningCandidateState] = []
        for existing in self.candidates:
            if (
                existing.context_key == candidate.context_key
                and existing.engine_setting_raw_value == candidate.engine_setting_raw_value
            ):
                candidates.append(candidate)
                replaced = True
            else:
                candidates.append(existing)
        if not replaced:
            candidates.append(candidate)
        candidates.sort(key=lambda item: (item.context_key, item.engine_setting_raw_value))
        return EngineTuningRuntimeState(
            version=self.version,
            update_count=self.update_count + 1,
            candidates=tuple(candidates),
            objective=self.objective,
            reward_fingerprint=self.reward_fingerprint,
            model_state=self.model_state,
        )

    def decay(self, stat_decay: float) -> EngineTuningRuntimeState:
        """Return state after discounting all model statistics once."""

        if not self.candidates:
            return self
        return replace(
            self,
            candidates=tuple(candidate.decay(stat_decay) for candidate in self.candidates),
        )

    def with_model_state(
        self,
        model_state: EngineTuningModelState | None,
        *,
        increment_update_count: bool = False,
    ) -> EngineTuningRuntimeState:
        """Return state with updated learned model weights."""

        return replace(
            self,
            update_count=self.update_count + (1 if increment_update_count else 0),
            model_state=model_state,
        )


def empty_engine_tuning_state() -> EngineTuningRuntimeState:
    """Return an empty state snapshot."""

    return EngineTuningRuntimeState(
        version=ENGINE_TUNING_STATE_VERSION,
        update_count=0,
        candidates=(),
        objective="finish_time",
        reward_fingerprint=None,
        model_state=None,
    )


def engine_tuning_state_with_objective(
    state: EngineTuningRuntimeState,
    *,
    objective: EngineTunerObjective,
    reward_fingerprint: str | None = None,
) -> EngineTuningRuntimeState:
    """Return a state whose active aggregates match the requested objective."""

    reward_changed = (
        state.reward_fingerprint is not None and state.reward_fingerprint != reward_fingerprint
    )
    if reward_changed:
        state = replace(
            state,
            candidates=tuple(
                candidate.without_return_observations() for candidate in state.candidates
            ),
        )
    if objective == "episode_return" and state.reward_fingerprint != reward_fingerprint:
        return empty_engine_tuning_state_for(
            objective=objective,
            reward_fingerprint=reward_fingerprint,
        )
    if state.objective == objective and (
        objective != "episode_return" or state.reward_fingerprint == reward_fingerprint
    ):
        return replace(
            state,
            version=ENGINE_TUNING_STATE_VERSION,
            reward_fingerprint=reward_fingerprint,
        )
    candidates = tuple(
        candidate
        for candidate in (item.with_active_objective(objective) for item in state.candidates)
        if candidate is not None
    )
    return EngineTuningRuntimeState(
        version=ENGINE_TUNING_STATE_VERSION,
        update_count=state.update_count,
        candidates=tuple(candidates),
        objective=objective,
        reward_fingerprint=reward_fingerprint,
        model_state=None,
    )


def empty_engine_tuning_state_for(
    *,
    objective: EngineTunerObjective,
    reward_fingerprint: str | None = None,
) -> EngineTuningRuntimeState:
    """Return an empty state with active objective metadata."""

    return EngineTuningRuntimeState(
        version=ENGINE_TUNING_STATE_VERSION,
        update_count=0,
        candidates=(),
        objective=objective,
        reward_fingerprint=reward_fingerprint,
        model_state=None,
    )
