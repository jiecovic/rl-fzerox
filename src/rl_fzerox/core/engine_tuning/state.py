# src/rl_fzerox/core/engine_tuning/state.py
"""Runtime state for adaptive engine tuning."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from rl_fzerox.core.engine_tuning.types import EngineTunerBackend

ENGINE_TUNING_STATE_VERSION = 5


@dataclass(frozen=True, slots=True)
class EngineTuningCandidateState:
    """Successful finish-time observations for one engine value in one context."""

    context_key: str
    course_key: str
    vehicle_id: str
    engine_setting_raw_value: int
    finish_count: int = 0
    decayed_count: float = 0.0
    decayed_score_total: float = 0.0
    score_total: float = 0.0
    best_score: float | None = None
    best_time_ms: int | None = None

    @property
    def mean_score(self) -> float | None:
        if self.decayed_count <= 0.0:
            return None
        return self.decayed_score_total / self.decayed_count

    @property
    def raw_mean_score(self) -> float | None:
        if self.finish_count <= 0:
            return None
        return self.score_total / self.finish_count

    def record(
        self,
        *,
        score: float,
        finish_time_ms: int,
    ) -> EngineTuningCandidateState:
        """Return state after one successful finish observation."""

        clamped_finish_time_ms = max(1, int(finish_time_ms))
        return replace(
            self,
            finish_count=self.finish_count + 1,
            decayed_count=self.decayed_count + 1.0,
            decayed_score_total=self.decayed_score_total + float(score),
            score_total=self.score_total + float(score),
            best_score=(
                float(score) if self.best_score is None else max(self.best_score, float(score))
            ),
            best_time_ms=(
                clamped_finish_time_ms
                if self.best_time_ms is None
                else min(self.best_time_ms, clamped_finish_time_ms)
            ),
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
        model_state=None,
    )
