# src/rl_fzerox/core/engine_tuning/state/runtime.py
"""Runtime snapshots for adaptive engine tuning."""

from __future__ import annotations

from dataclasses import dataclass, replace

from rl_fzerox.core.engine_tuning.state.candidate import EngineTuningCandidateState
from rl_fzerox.core.engine_tuning.state.model import EngineTuningModelState
from rl_fzerox.core.engine_tuning.types import EngineTunerObjective

ENGINE_TUNING_STATE_VERSION = 7


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
    safe_finish_rate_threshold: float = 0.9,
    prior_finish_time_seconds: float = 200.0,
) -> EngineTuningRuntimeState:
    """Return a state whose active aggregates match the requested objective."""

    candidates = tuple(
        candidate
        for candidate in (
            item.with_active_objective(
                objective,
                safe_finish_rate_threshold=safe_finish_rate_threshold,
                prior_finish_time_seconds=prior_finish_time_seconds,
            )
            for item in state.candidates
        )
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
