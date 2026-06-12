# src/rl_fzerox/core/engine_tuning/tuner.py
"""Public adaptive engine tuner facade."""

from __future__ import annotations

from rl_fzerox.core.engine_tuning.ensemble import MlpEnsembleEngineTuner
from rl_fzerox.core.engine_tuning.gaussian_process import GaussianProcessEngineTuner
from rl_fzerox.core.engine_tuning.state import (
    EngineTuningRuntimeState,
    empty_engine_tuning_state,
)
from rl_fzerox.core.engine_tuning.types import (
    EngineTunerBackend,
    EngineTunerSettings,
    EngineTuningCandidateEstimate,
    EngineTuningChoice,
    EngineTuningContext,
    EngineTuningEpisodeOutcome,
    engine_candidates,
    finish_time_ms_from_score,
    finish_time_score,
)


class OrderedEngineTuner:
    """Choose engine values with the configured ordered tuner backend."""

    def __init__(
        self,
        *,
        settings: EngineTunerSettings,
        state: EngineTuningRuntimeState | None = None,
    ) -> None:
        self._settings = settings
        self._backend = _backend_for(settings=settings, state=state)

    @property
    def state(self) -> EngineTuningRuntimeState:
        return self._backend.state

    def choose(self, context: EngineTuningContext, *, seed: int | None) -> EngineTuningChoice:
        """Sample one integer engine value for the given context."""

        return self._backend.choose(context, seed=seed)

    def recommendation(self, context: EngineTuningContext) -> EngineTuningChoice:
        """Return the greedy engine value for the given context."""

        return self._backend.recommendation(context)

    def distribution(
        self,
        context: EngineTuningContext,
        *,
        seed: int,
        draws: int = 512,
    ) -> tuple[EngineTuningCandidateEstimate, ...]:
        """Estimate the current stochastic reset distribution for one context."""

        return self._backend.distribution(context, seed=seed, draws=draws)

    def record(self, outcome: EngineTuningEpisodeOutcome) -> EngineTuningRuntimeState:
        """Update the active backend from one terminal episode result."""

        return self.record_many((outcome,))

    def record_many(
        self,
        outcomes: tuple[EngineTuningEpisodeOutcome, ...],
    ) -> EngineTuningRuntimeState:
        """Update the active backend from one rollout batch."""

        return self._backend.record_many(outcomes)

    def score(self, outcome: EngineTuningEpisodeOutcome) -> float:
        """Return the backend score for one episode outcome."""

        return self._backend.score(outcome)


def _backend_for(
    *,
    settings: EngineTunerSettings,
    state: EngineTuningRuntimeState | None,
) -> GaussianProcessEngineTuner | MlpEnsembleEngineTuner:
    runtime_state = state or empty_engine_tuning_state()
    if settings.backend == "mlp_ensemble":
        return MlpEnsembleEngineTuner(settings=settings, state=runtime_state)
    return GaussianProcessEngineTuner(settings=settings, state=runtime_state)


__all__ = (
    "EngineTunerSettings",
    "EngineTunerBackend",
    "EngineTuningCandidateEstimate",
    "EngineTuningChoice",
    "EngineTuningContext",
    "EngineTuningEpisodeOutcome",
    "OrderedEngineTuner",
    "engine_candidates",
    "finish_time_ms_from_score",
    "finish_time_score",
)
