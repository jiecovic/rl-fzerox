# src/rl_fzerox/core/engine_tuning/tuner.py
"""Ordered finish-time tuner for reset-time engine settings."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from random import Random

import gpytorch
import torch

from rl_fzerox.core.engine_tuning.state import (
    EngineTuningCandidateState,
    EngineTuningRuntimeState,
    empty_engine_tuning_state,
)


@dataclass(frozen=True, slots=True)
class EngineTunerDefaults:
    """Default scale values for the finish-time GP tuner."""

    prior_finish_time_seconds: float = 200.0
    exploration_seconds: float = 30.0
    observation_noise_seconds: float = 1.5
    curve_lengthscale_raw: float = 12.0


ENGINE_TUNER_DEFAULTS = EngineTunerDefaults()


@dataclass(frozen=True, slots=True)
class EngineTunerSettings:
    """Static knobs for one adaptive engine-tuning run."""

    min_raw_value: int = 0
    max_raw_value: int = 100
    stat_decay: float = 0.99
    prior_finish_time_seconds: float = ENGINE_TUNER_DEFAULTS.prior_finish_time_seconds
    exploration_seconds: float = ENGINE_TUNER_DEFAULTS.exploration_seconds
    observation_noise_seconds: float = ENGINE_TUNER_DEFAULTS.observation_noise_seconds
    curve_lengthscale_raw: float = ENGINE_TUNER_DEFAULTS.curve_lengthscale_raw
    uniform_exploration: float = 0.05


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
    finish_count: int
    estimated_finish_time_ms: int
    best_finish_time_ms: int | None


@dataclass(frozen=True, slots=True)
class EngineTuningCandidateEstimate:
    """Estimated reset-time selection probability for one engine value."""

    engine_setting_raw_value: int
    probability: float
    posterior_mean: float
    estimated_finish_time_ms: int
    finish_count: int
    best_finish_time_ms: int | None


@dataclass(frozen=True, slots=True)
class EngineTuningEpisodeOutcome:
    """Episode result used to score one successful engine-setting sample."""

    context: EngineTuningContext
    engine_setting_raw_value: int
    completion_fraction: float
    finished: bool
    race_time_ms: int | None = None
    finish_position: int | None = None
    total_racers: int | None = None


class OrderedEngineTuner:
    """Choose engine values from a smooth finish-time model over the 0-100 slider."""

    def __init__(
        self,
        *,
        settings: EngineTunerSettings,
        state: EngineTuningRuntimeState | None = None,
    ) -> None:
        self._settings = settings
        self._state = state or empty_engine_tuning_state()

    @property
    def state(self) -> EngineTuningRuntimeState:
        return self._state

    def choose(self, context: EngineTuningContext, *, seed: int | None) -> EngineTuningChoice:
        """Sample one integer engine value for the given context."""

        rng = Random(seed) if seed is not None else Random()
        candidates = engine_candidates(
            minimum=self._settings.min_raw_value,
            maximum=self._settings.max_raw_value,
        )
        if rng.random() < max(0.0, min(1.0, self._settings.uniform_exploration)):
            projection = self._context_projection(context, candidates)
            selected = rng.choice(candidates)
            return self._choice_for(
                context,
                selected,
                estimate=projection.estimates[selected],
                sampled_score=None,
            )

        best: EngineTuningChoice | None = None
        projection = self._context_projection(context, candidates)
        sampled_scores = _sample_posterior_scores(
            projection=projection,
            candidates=candidates,
            rng=rng,
        )
        for engine_raw, sampled_score in zip(candidates, sampled_scores, strict=True):
            estimate = projection.estimates[engine_raw]
            choice = self._choice_for(
                context,
                engine_raw,
                estimate=estimate,
                sampled_score=sampled_score,
            )
            if best is None or _better_choice(choice, best, candidates=candidates):
                best = choice
        if best is None:
            raise ValueError("adaptive engine tuning has no engine candidates")
        return best

    def recommendation(self, context: EngineTuningContext) -> EngineTuningChoice:
        """Return the lowest predicted finish-time value without random exploration."""

        best: EngineTuningChoice | None = None
        candidates = engine_candidates(
            minimum=self._settings.min_raw_value,
            maximum=self._settings.max_raw_value,
        )
        estimates = self._context_projection(context, candidates).estimates
        for engine_raw in candidates:
            estimate = estimates[engine_raw]
            choice = self._choice_for(
                context,
                engine_raw,
                estimate=estimate,
                sampled_score=None,
            )
            if best is None or _better_choice(choice, best, candidates=candidates):
                best = choice
        if best is None:
            raise ValueError("adaptive engine tuning has no engine candidates")
        return best

    def distribution(
        self,
        context: EngineTuningContext,
        *,
        seed: int,
        draws: int = 512,
    ) -> tuple[EngineTuningCandidateEstimate, ...]:
        """Estimate the current stochastic reset distribution for one context."""

        candidates = engine_candidates(
            minimum=self._settings.min_raw_value,
            maximum=self._settings.max_raw_value,
        )
        if not candidates:
            raise ValueError("adaptive engine tuning has no engine candidates")

        draw_count = max(1, int(draws))
        counts = dict.fromkeys(candidates, 0)
        rng = Random(seed)
        projection = self._context_projection(context, candidates)
        for _ in range(draw_count):
            best_raw: int | None = None
            best_score: float | None = None
            sampled_scores = _sample_posterior_scores(
                projection=projection,
                candidates=candidates,
                rng=rng,
            )
            for engine_raw, sampled_score in zip(candidates, sampled_scores, strict=True):
                if best_score is None or sampled_score > best_score:
                    best_raw = engine_raw
                    best_score = sampled_score
            if best_raw is not None:
                counts[best_raw] += 1

        uniform_probability = max(0.0, min(1.0, self._settings.uniform_exploration)) / len(
            candidates
        )
        sample_probability_scale = 1.0 - max(
            0.0,
            min(1.0, self._settings.uniform_exploration),
        )
        return tuple(
            EngineTuningCandidateEstimate(
                engine_setting_raw_value=engine_raw,
                probability=uniform_probability
                + sample_probability_scale * (counts[engine_raw] / draw_count),
                posterior_mean=projection.estimates[engine_raw].posterior_mean,
                estimated_finish_time_ms=finish_time_ms_from_score(
                    projection.estimates[engine_raw].posterior_mean
                ),
                finish_count=projection.estimates[engine_raw].exact_finish_count,
                best_finish_time_ms=projection.estimates[engine_raw].best_finish_time_ms,
            )
            for engine_raw in candidates
        )

    def record(self, outcome: EngineTuningEpisodeOutcome) -> EngineTuningRuntimeState:
        """Update the state from one terminal episode result."""

        finish_time_ms = _successful_finish_time_ms(outcome)
        if finish_time_ms is None:
            return self._state
        score = finish_time_score(finish_time_ms)
        decayed_state = self._state.decay(self._settings.stat_decay)
        candidate = _candidate_from_state(
            decayed_state,
            outcome.context,
            outcome.engine_setting_raw_value,
        ).record(
            score=score,
            finish_time_ms=finish_time_ms,
        )
        self._state = decayed_state.with_candidate(candidate)
        return self._state

    def score(self, outcome: EngineTuningEpisodeOutcome) -> float:
        """Return a higher-is-better negative finish-time score."""

        finish_time_ms = _successful_finish_time_ms(outcome)
        if finish_time_ms is None:
            return self._prior_score()
        return finish_time_score(finish_time_ms)

    def _choice_for(
        self,
        context: EngineTuningContext,
        engine_raw: int,
        *,
        estimate: _EngineEstimate,
        sampled_score: float | None,
    ) -> EngineTuningChoice:
        exact_candidate = self._state.candidate_map().get((context.key, int(engine_raw)))
        return EngineTuningChoice(
            context=context,
            engine_setting_raw_value=engine_raw,
            sampled_score=estimate.posterior_mean if sampled_score is None else sampled_score,
            mean_score=estimate.posterior_mean,
            finish_count=0 if exact_candidate is None else exact_candidate.finish_count,
            estimated_finish_time_ms=finish_time_ms_from_score(estimate.posterior_mean),
            best_finish_time_ms=None if exact_candidate is None else exact_candidate.best_time_ms,
        )

    def _context_projection(
        self,
        context: EngineTuningContext,
        candidates: tuple[int, ...],
    ) -> _EngineProjection:
        observed_candidates = tuple(
            candidate
            for candidate in self._state.candidates
            if candidate.context_key == context.key and candidate.finish_count > 0
        )
        if not observed_candidates:
            return _EngineProjection(
                estimates={
                    engine_raw: self._prior_estimate(context=context, engine_raw=engine_raw)
                    for engine_raw in candidates
                },
                covariance=None,
            )

        posterior = _gp_posterior(
            observed_candidates=observed_candidates,
            candidate_raw_values=candidates,
            prior_score=self._prior_score(),
            settings=self._settings,
        )
        candidate_map = self._state.candidate_map()
        estimates: dict[int, _EngineEstimate] = {}
        for index, engine_raw in enumerate(candidates):
            candidate = candidate_map.get((context.key, int(engine_raw)))
            estimates[engine_raw] = _EngineEstimate(
                posterior_mean=posterior.means[index],
                posterior_std=posterior.stds[index],
                exact_finish_count=0 if candidate is None else candidate.finish_count,
                best_finish_time_ms=None if candidate is None else candidate.best_time_ms,
            )
        return _EngineProjection(estimates=estimates, covariance=posterior.covariance)

    def _prior_estimate(
        self,
        *,
        context: EngineTuningContext,
        engine_raw: int,
    ) -> _EngineEstimate:
        candidate = self._state.candidate_map().get((context.key, int(engine_raw)))
        return _EngineEstimate(
            posterior_mean=self._prior_score(),
            posterior_std=max(0.0, float(self._settings.exploration_seconds)),
            exact_finish_count=0 if candidate is None else candidate.finish_count,
            best_finish_time_ms=None if candidate is None else candidate.best_time_ms,
        )

    def _prior_score(self) -> float:
        return -max(1.0, float(self._settings.prior_finish_time_seconds))


@dataclass(frozen=True, slots=True)
class _EngineEstimate:
    posterior_mean: float
    posterior_std: float
    exact_finish_count: int
    best_finish_time_ms: int | None


@dataclass(frozen=True, slots=True)
class _EngineProjection:
    estimates: dict[int, _EngineEstimate]
    covariance: torch.Tensor | None


@dataclass(frozen=True, slots=True)
class _EnginePosterior:
    means: tuple[float, ...]
    stds: tuple[float, ...]
    covariance: torch.Tensor


class _EngineGPModel(gpytorch.models.ExactGP):
    def __init__(
        self,
        *,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods.FixedNoiseGaussianLikelihood,
        prior_score: float,
        lengthscale: float,
        outputscale: float,
    ) -> None:
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.mean_module.constant = torch.as_tensor(prior_score, dtype=train_x.dtype)
        self.covar_module.base_kernel.lengthscale = torch.as_tensor(
            lengthscale,
            dtype=train_x.dtype,
        )
        self.covar_module.outputscale = torch.as_tensor(outputscale, dtype=train_x.dtype)
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        mean = self.mean_module(x)
        covariance = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covariance)


def engine_candidates(*, minimum: int, maximum: int) -> tuple[int, ...]:
    """Return inclusive integer engine values clamped to the game's raw 0-100 range."""

    lower = max(0, min(100, int(minimum)))
    upper = max(0, min(100, int(maximum)))
    if lower > upper:
        raise ValueError(f"engine tuning min_raw_value exceeds max_raw_value: {lower} > {upper}")
    return tuple(range(lower, upper + 1))


def _candidate_from_state(
    state: EngineTuningRuntimeState,
    context: EngineTuningContext,
    engine_raw: int,
) -> EngineTuningCandidateState:
    candidate = state.candidate_map().get((context.key, int(engine_raw)))
    if candidate is not None:
        return candidate
    return EngineTuningCandidateState(
        context_key=context.key,
        course_key=context.course_key,
        vehicle_id=context.vehicle_id,
        engine_setting_raw_value=int(engine_raw),
    )


def finish_time_score(race_time_ms: int) -> float:
    """Return a higher-is-better score in negative seconds."""

    return -(max(1.0, float(race_time_ms)) * 0.001)


def finish_time_ms_from_score(score: float) -> int:
    """Return a positive finish-time estimate from a negative-seconds score."""

    return max(1, int(round(max(0.001, -float(score)) * 1000.0)))


def _successful_finish_time_ms(outcome: EngineTuningEpisodeOutcome) -> int | None:
    if not outcome.finished or outcome.race_time_ms is None or outcome.race_time_ms <= 0:
        return None
    return int(outcome.race_time_ms)


def _smoothing_bandwidth(settings: EngineTunerSettings) -> float:
    return max(1.0, float(settings.curve_lengthscale_raw))


def _gp_posterior(
    *,
    observed_candidates: tuple[EngineTuningCandidateState, ...],
    candidate_raw_values: tuple[int, ...],
    prior_score: float,
    settings: EngineTunerSettings,
) -> _EnginePosterior:
    train_x = torch.as_tensor(
        [
            [_normalize_engine_raw(candidate.engine_setting_raw_value)]
            for candidate in observed_candidates
        ],
        dtype=torch.float64,
    )
    train_y = torch.as_tensor(
        [
            candidate.mean_score if candidate.mean_score is not None else prior_score
            for candidate in observed_candidates
        ],
        dtype=torch.float64,
    )
    train_noise = torch.as_tensor(
        [
            _observation_noise_variance(candidate, settings=settings)
            for candidate in observed_candidates
        ],
        dtype=torch.float64,
    )
    likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
        noise=train_noise,
        learn_additional_noise=False,
    )
    model = _EngineGPModel(
        train_x=train_x,
        train_y=train_y,
        likelihood=likelihood,
        prior_score=prior_score,
        lengthscale=_smoothing_bandwidth(settings) / 100.0,
        outputscale=max(1.0, float(settings.exploration_seconds)) ** 2,
    )
    model.eval()
    likelihood.eval()
    test_x = torch.as_tensor(
        [[_normalize_engine_raw(raw_value)] for raw_value in candidate_raw_values],
        dtype=torch.float64,
    )
    with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.debug(False):
        posterior = model(test_x)
    means = tuple(float(value) for value in posterior.mean.detach().cpu().tolist())
    variances = posterior.variance.detach().clamp_min(1e-9).cpu().tolist()
    stds = tuple(sqrt(float(value)) for value in variances)
    covariance = posterior.covariance_matrix.detach().cpu()
    return _EnginePosterior(means=means, stds=stds, covariance=covariance)


def _sample_posterior_scores(
    *,
    projection: _EngineProjection,
    candidates: tuple[int, ...],
    rng: Random,
) -> tuple[float, ...]:
    means = torch.as_tensor(
        [projection.estimates[engine_raw].posterior_mean for engine_raw in candidates],
        dtype=torch.float64,
    )
    if projection.covariance is None:
        stds = [projection.estimates[engine_raw].posterior_std for engine_raw in candidates]
        return tuple(rng.gauss(mean, std) for mean, std in zip(means.tolist(), stds, strict=True))

    covariance = projection.covariance.to(dtype=torch.float64)
    jitter = torch.eye(covariance.shape[0], dtype=torch.float64) * 1e-9
    cholesky = torch.linalg.cholesky(covariance + jitter)
    standard_normals = torch.as_tensor(
        [rng.gauss(0.0, 1.0) for _ in candidates],
        dtype=torch.float64,
    )
    sample = means + cholesky @ standard_normals
    return tuple(float(value) for value in sample.tolist())


def _better_choice(
    candidate: EngineTuningChoice,
    current: EngineTuningChoice,
    *,
    candidates: tuple[int, ...],
) -> bool:
    if candidate.sampled_score != current.sampled_score:
        return candidate.sampled_score > current.sampled_score
    midpoint = (candidates[0] + candidates[-1]) / 2.0
    candidate_distance = abs(candidate.engine_setting_raw_value - midpoint)
    current_distance = abs(current.engine_setting_raw_value - midpoint)
    if candidate_distance != current_distance:
        return candidate_distance < current_distance
    return candidate.engine_setting_raw_value < current.engine_setting_raw_value


def _normalize_engine_raw(raw_value: int) -> float:
    return max(0.0, min(1.0, float(raw_value) / 100.0))


def _observation_noise_variance(
    candidate: EngineTuningCandidateState,
    *,
    settings: EngineTunerSettings,
) -> float:
    base_std = max(0.25, float(settings.observation_noise_seconds))
    effective_count = max(1e-6, float(candidate.decayed_count))
    return (base_std * base_std) / effective_count
