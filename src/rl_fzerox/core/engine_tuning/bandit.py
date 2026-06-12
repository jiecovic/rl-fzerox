# src/rl_fzerox/core/engine_tuning/bandit.py
"""Discounted Thompson sampler for reset-time engine settings."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from random import Random

from rl_fzerox.core.engine_tuning.state import (
    EngineTuningArmState,
    EngineTuningRuntimeState,
    empty_engine_tuning_state,
)


@dataclass(frozen=True, slots=True)
class EngineBanditSettings:
    """Static knobs for one adaptive engine-tuning run."""

    min_raw_value: int = 0
    max_raw_value: int = 100
    bin_size: int = 5
    stat_decay: float = 0.99
    prior_mean: float = 0.5
    prior_strength: float = 2.0
    exploration_scale: float = 0.35
    uniform_exploration: float = 0.05
    completion_weight: float = 1.0
    finish_bonus: float = 1.0
    position_weight: float = 0.25


@dataclass(frozen=True, slots=True)
class EngineTuningContext:
    """Stable identity for a family of engine-setting attempts."""

    course_key: str
    vehicle_id: str

    @property
    def key(self) -> str:
        return f"{self.course_key}|{self.vehicle_id}"


@dataclass(frozen=True, slots=True)
class EngineTuningChoice:
    """One reset-time engine choice plus diagnostic fields."""

    context: EngineTuningContext
    engine_setting_raw_value: int
    sampled_score: float
    mean_score: float
    attempts: int


@dataclass(frozen=True, slots=True)
class EngineTuningBinProbability:
    """Estimated reset-time selection probability for one engine bin."""

    engine_setting_raw_value: int
    probability: float
    posterior_mean: float
    attempts: int


@dataclass(frozen=True, slots=True)
class EngineTuningEpisodeOutcome:
    """Episode result used to score one engine-setting attempt."""

    context: EngineTuningContext
    engine_setting_raw_value: int
    completion_fraction: float
    finished: bool
    finish_position: int | None = None
    total_racers: int | None = None


class AdaptiveEngineBandit:
    """Choose and update engine bins using discounted Thompson sampling."""

    def __init__(
        self,
        *,
        settings: EngineBanditSettings,
        state: EngineTuningRuntimeState | None = None,
    ) -> None:
        self._settings = settings
        self._state = state or empty_engine_tuning_state()

    @property
    def state(self) -> EngineTuningRuntimeState:
        return self._state

    def choose(self, context: EngineTuningContext, *, seed: int | None) -> EngineTuningChoice:
        """Sample one engine bin for the given context."""

        rng = Random(seed) if seed is not None else Random()
        bins = engine_bins(
            minimum=self._settings.min_raw_value,
            maximum=self._settings.max_raw_value,
            bin_size=self._settings.bin_size,
        )
        if rng.random() < max(0.0, min(1.0, self._settings.uniform_exploration)):
            engine_raw = rng.choice(bins)
            arm = self._arm(context, engine_raw)
            mean = self._posterior_mean(arm)
            return EngineTuningChoice(
                context=context,
                engine_setting_raw_value=engine_raw,
                sampled_score=mean,
                mean_score=mean,
                attempts=arm.attempts,
            )

        best: EngineTuningChoice | None = None
        for engine_raw in bins:
            arm = self._arm(context, engine_raw)
            mean = self._posterior_mean(arm)
            sampled_score = rng.gauss(mean, self._posterior_std(arm))
            choice = EngineTuningChoice(
                context=context,
                engine_setting_raw_value=engine_raw,
                sampled_score=sampled_score,
                mean_score=mean,
                attempts=arm.attempts,
            )
            if best is None or choice.sampled_score > best.sampled_score:
                best = choice
        if best is None:
            raise ValueError("adaptive engine tuning has no engine bins")
        return best

    def recommendation(self, context: EngineTuningContext) -> EngineTuningChoice:
        """Return the highest posterior-mean bin without random exploration."""

        best: EngineTuningChoice | None = None
        for engine_raw in engine_bins(
            minimum=self._settings.min_raw_value,
            maximum=self._settings.max_raw_value,
            bin_size=self._settings.bin_size,
        ):
            arm = self._arm(context, engine_raw)
            mean = self._posterior_mean(arm)
            choice = EngineTuningChoice(
                context=context,
                engine_setting_raw_value=engine_raw,
                sampled_score=mean,
                mean_score=mean,
                attempts=arm.attempts,
            )
            if best is None or choice.mean_score > best.mean_score:
                best = choice
        if best is None:
            raise ValueError("adaptive engine tuning has no engine bins")
        return best

    def choice_distribution(
        self,
        context: EngineTuningContext,
        *,
        seed: int,
        draws: int = 512,
    ) -> tuple[EngineTuningBinProbability, ...]:
        """Estimate the current stochastic reset distribution for one context."""

        bins = engine_bins(
            minimum=self._settings.min_raw_value,
            maximum=self._settings.max_raw_value,
            bin_size=self._settings.bin_size,
        )
        if not bins:
            raise ValueError("adaptive engine tuning has no engine bins")

        draw_count = max(1, int(draws))
        counts = dict.fromkeys(bins, 0)
        rng = Random(seed)
        for _ in range(draw_count):
            best_raw: int | None = None
            best_score: float | None = None
            for engine_raw in bins:
                arm = self._arm(context, engine_raw)
                sampled_score = rng.gauss(self._posterior_mean(arm), self._posterior_std(arm))
                if best_score is None or sampled_score > best_score:
                    best_raw = engine_raw
                    best_score = sampled_score
            if best_raw is not None:
                counts[best_raw] += 1

        uniform_probability = max(0.0, min(1.0, self._settings.uniform_exploration)) / len(bins)
        thompson_probability_scale = 1.0 - max(
            0.0,
            min(1.0, self._settings.uniform_exploration),
        )
        return tuple(
            EngineTuningBinProbability(
                engine_setting_raw_value=engine_raw,
                probability=uniform_probability
                + thompson_probability_scale * (counts[engine_raw] / draw_count),
                posterior_mean=self._posterior_mean(self._arm(context, engine_raw)),
                attempts=self._arm(context, engine_raw).attempts,
            )
            for engine_raw in bins
        )

    def record(self, outcome: EngineTuningEpisodeOutcome) -> EngineTuningRuntimeState:
        """Update the state from one terminal episode result."""

        score = self.score(outcome)
        arm = self._arm(outcome.context, outcome.engine_setting_raw_value).record(
            score=score,
            completion_fraction=outcome.completion_fraction,
            finished=outcome.finished,
            stat_decay=self._settings.stat_decay,
        )
        self._state = self._state.with_arm(arm)
        return self._state

    def score(self, outcome: EngineTuningEpisodeOutcome) -> float:
        """Return a scalar score for one episode outcome."""

        completion = max(0.0, min(1.0, float(outcome.completion_fraction)))
        position_score = _position_score(outcome.finish_position, outcome.total_racers)
        return (
            max(0.0, self._settings.completion_weight) * completion
            + (max(0.0, self._settings.finish_bonus) if outcome.finished else 0.0)
            + max(0.0, self._settings.position_weight) * position_score
        )

    def _arm(self, context: EngineTuningContext, engine_raw: int) -> EngineTuningArmState:
        arm = self._state.arm_map().get((context.key, int(engine_raw)))
        if arm is not None:
            return arm
        return EngineTuningArmState(
            context_key=context.key,
            course_key=context.course_key,
            vehicle_id=context.vehicle_id,
            engine_setting_raw_value=int(engine_raw),
        )

    def _posterior_mean(self, arm: EngineTuningArmState) -> float:
        mean = arm.mean_score
        if mean is None:
            return float(self._settings.prior_mean)
        prior_strength = max(0.0, float(self._settings.prior_strength))
        return (mean * arm.decayed_count + self._settings.prior_mean * prior_strength) / max(
            arm.decayed_count + prior_strength, 1e-9
        )

    def _posterior_std(self, arm: EngineTuningArmState) -> float:
        count = max(0.0, arm.decayed_count) + max(0.0, self._settings.prior_strength)
        return max(0.0, self._settings.exploration_scale) / sqrt(max(count, 1.0))


def engine_bins(*, minimum: int, maximum: int, bin_size: int) -> tuple[int, ...]:
    """Return inclusive engine bins clamped to the game's raw 0-100 range."""

    lower = max(0, min(100, int(minimum)))
    upper = max(0, min(100, int(maximum)))
    if lower > upper:
        raise ValueError(f"engine tuning min_raw_value exceeds max_raw_value: {lower} > {upper}")
    step = max(1, int(bin_size))
    bins = tuple(range(lower, upper + 1, step))
    if not bins or bins[-1] != upper:
        bins = (*bins, upper)
    return bins


def _position_score(position: int | None, total_racers: int | None) -> float:
    if position is None or total_racers is None or total_racers <= 1:
        return 0.0
    clamped_position = max(1, min(int(position), int(total_racers)))
    return (total_racers - clamped_position) / (total_racers - 1)
