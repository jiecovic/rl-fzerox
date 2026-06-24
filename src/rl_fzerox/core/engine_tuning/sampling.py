# src/rl_fzerox/core/engine_tuning/sampling.py
"""Reset-time engine-tuning sampler snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Literal

from rl_fzerox.core.engine_tuning.types import (
    EngineTuningChoice,
    EngineTuningContext,
)

EngineTuningSelectionMode = Literal["sample", "greedy"]


@dataclass(frozen=True, slots=True)
class EngineTuningResetCandidate:
    """One immutable engine candidate available to reset workers."""

    engine_setting_raw_value: int
    probability: float
    mean_score: float
    uncertainty_score: float
    score_count: int
    finish_count: int
    estimated_finish_time_ms: int
    best_finish_time_ms: int | None

    def choice(self, context: EngineTuningContext) -> EngineTuningChoice:
        """Return the env-facing choice diagnostics for this candidate."""

        # Reset workers sample from cached probabilities, not from the tuner
        # posterior, so no fresh posterior score sample exists here.
        return EngineTuningChoice(
            context=context,
            engine_setting_raw_value=self.engine_setting_raw_value,
            sampled_score=self.mean_score,
            mean_score=self.mean_score,
            uncertainty_score=self.uncertainty_score,
            score_count=self.score_count,
            finish_count=self.finish_count,
            estimated_finish_time_ms=self.estimated_finish_time_ms,
            best_finish_time_ms=self.best_finish_time_ms,
        )


@dataclass(frozen=True, slots=True)
class EngineTuningResetContext:
    """All reset-time candidates for one course and vehicle."""

    context: EngineTuningContext
    candidates: tuple[EngineTuningResetCandidate, ...]
    greedy_engine_setting_raw_value: int

    def choose(
        self,
        *,
        selection: EngineTuningSelectionMode,
        seed: int | None,
    ) -> EngineTuningChoice | None:
        """Resolve one reset-time engine value without loading tuner state."""

        if not self.candidates:
            return None
        if selection == "greedy":
            return self._greedy_candidate().choice(self.context)
        return self._sample_candidate(seed=seed).choice(self.context)

    def _greedy_candidate(self) -> EngineTuningResetCandidate:
        for candidate in self.candidates:
            if candidate.engine_setting_raw_value == self.greedy_engine_setting_raw_value:
                return candidate
        return max(self.candidates, key=lambda candidate: candidate.probability)

    def _sample_candidate(self, *, seed: int | None) -> EngineTuningResetCandidate:
        total_probability = sum(max(0.0, candidate.probability) for candidate in self.candidates)
        if total_probability <= 0.0:
            return self._greedy_candidate()
        rng = Random(seed) if seed is not None else Random()
        sample = rng.random() * total_probability
        cursor = 0.0
        for candidate in self.candidates:
            cursor += max(0.0, candidate.probability)
            if sample <= cursor:
                return candidate
        return self.candidates[-1]


@dataclass(frozen=True, slots=True)
class EngineTuningResetSampler:
    """Plain reset-worker snapshot of adaptive engine choices."""

    contexts: tuple[EngineTuningResetContext, ...]

    def context(self, context: EngineTuningContext) -> EngineTuningResetContext | None:
        """Return the reset candidates for one course and vehicle."""

        key = context.key
        for reset_context in self.contexts:
            if reset_context.context.key == key:
                return reset_context
        return None

    def choose(
        self,
        context: EngineTuningContext,
        *,
        selection: EngineTuningSelectionMode,
        seed: int | None,
    ) -> EngineTuningChoice | None:
        """Resolve one engine choice from this snapshot."""

        reset_context = self.context(context)
        if reset_context is None:
            return None
        return reset_context.choose(selection=selection, seed=seed)
