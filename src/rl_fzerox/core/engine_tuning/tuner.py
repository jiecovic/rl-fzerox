# src/rl_fzerox/core/engine_tuning/tuner.py
"""Public adaptive engine tuner facade."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rl_fzerox.core.engine_tuning.bandit import BanditEngineTuner
from rl_fzerox.core.engine_tuning.state import (
    EngineTuningRuntimeState,
    empty_engine_tuning_state,
)
from rl_fzerox.core.engine_tuning.types import (
    BanditEngineTunerSettings,
    EngineTunerBackend,
    EngineTunerSettings,
    EngineTuningCandidateEstimate,
    EngineTuningChoice,
    EngineTuningContext,
    EngineTuningEpisodeOutcome,
    GaussianProcessEngineTunerSettings,
    MlpEnsembleEngineTunerSettings,
    engine_candidates,
    finish_time_ms_from_score,
    finish_time_score,
)

if TYPE_CHECKING:
    from rl_fzerox.core.engine_tuning.experimental.ensemble import MlpEnsembleEngineTuner
    from rl_fzerox.core.engine_tuning.experimental.gaussian_process import (
        GaussianProcessEngineTuner,
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
) -> BanditEngineTuner | GaussianProcessEngineTuner | MlpEnsembleEngineTuner:
    runtime_state = state or empty_engine_tuning_state()
    if isinstance(settings, BanditEngineTunerSettings):
        return BanditEngineTuner(settings=settings, state=runtime_state)
    if isinstance(settings, MlpEnsembleEngineTunerSettings):
        from rl_fzerox.core.engine_tuning.experimental.ensemble import (
            MlpEnsembleEngineTuner,
        )

        return MlpEnsembleEngineTuner(settings=settings, state=runtime_state)
    if isinstance(settings, GaussianProcessEngineTunerSettings):
        from rl_fzerox.core.engine_tuning.experimental.gaussian_process import (
            GaussianProcessEngineTuner,
        )

        return GaussianProcessEngineTuner(settings=settings, state=runtime_state)
    raise TypeError(f"unsupported engine tuner settings: {type(settings).__name__}")


__all__ = (
    "EngineTunerSettings",
    "EngineTunerBackend",
    "EngineTuningCandidateEstimate",
    "EngineTuningChoice",
    "EngineTuningContext",
    "EngineTuningEpisodeOutcome",
    "BanditEngineTunerSettings",
    "GaussianProcessEngineTunerSettings",
    "MlpEnsembleEngineTunerSettings",
    "OrderedEngineTuner",
    "engine_candidates",
    "finish_time_ms_from_score",
    "finish_time_score",
)
