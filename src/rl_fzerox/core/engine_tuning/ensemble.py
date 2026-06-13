# src/rl_fzerox/core/engine_tuning/ensemble.py
"""Bootstrapped MLP ensemble backend for adaptive engine tuning."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from math import exp, sqrt
from random import Random

import torch

from rl_fzerox.core.engine_tuning.state import (
    EngineTuningEnsembleMemberState,
    EngineTuningModelContextState,
    EngineTuningModelState,
    EngineTuningRuntimeState,
    EngineTuningTensorState,
    empty_engine_tuning_state,
)
from rl_fzerox.core.engine_tuning.types import (
    EngineTuningCandidateEstimate,
    EngineTuningChoice,
    EngineTuningContext,
    EngineTuningEpisodeOutcome,
    MlpEnsembleEngineTunerSettings,
    engine_candidates,
    finish_time_ms_from_score,
    finish_time_score,
    successful_finish_time_ms,
)


@dataclass(frozen=True, slots=True)
class EngineTuningEnsembleShape:
    """Small fixed architecture for the engine-tuning surrogate."""

    member_count: int = 5
    course_embedding_dim: int = 8
    vehicle_embedding_dim: int = 4
    engine_basis_count: int = 21
    engine_basis_width_raw: float = 7.5
    hidden_dim: int = 32
    training_steps: int = 48
    learning_rate: float = 0.004
    bootstrap_keep_probability: float = 0.8
    sampling_warmup_successes: int = 32
    sampling_min_temperature_seconds: float = 2.0


ENGINE_TUNING_ENSEMBLE_SHAPE = EngineTuningEnsembleShape()
_SuccessfulOutcome = tuple[EngineTuningEpisodeOutcome, float, int]


class MlpEnsembleEngineTuner:
    """Choose engine values from a bootstrapped neural finish-time surrogate."""

    def __init__(
        self,
        *,
        settings: MlpEnsembleEngineTunerSettings,
        state: EngineTuningRuntimeState | None = None,
    ) -> None:
        self._settings = settings
        self._state = _mlp_state_or_empty(state)
        self._shape = replace(
            ENGINE_TUNING_ENSEMBLE_SHAPE,
            member_count=max(1, int(settings.ensemble_members)),
            hidden_dim=max(1, int(settings.hidden_dim)),
            training_steps=max(1, int(settings.training_steps)),
            learning_rate=max(1.0e-8, float(settings.learning_rate)),
            bootstrap_keep_probability=max(
                1.0e-6,
                min(1.0, float(settings.bootstrap_keep_probability)),
            ),
            sampling_warmup_successes=max(1, int(settings.warmup_successes)),
        )

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
        projection = self._context_projection(context, candidates)
        selected = _sample_candidate(
            candidates,
            probabilities=_sampling_probabilities(
                projection=projection,
                candidates=candidates,
                settings=self._settings,
                shape=self._shape,
            ),
            rng=rng,
        )
        return self._choice_for(
            context,
            selected,
            estimate=projection.estimates[selected],
            sampled_score=None,
        )

    def recommendation(self, context: EngineTuningContext) -> EngineTuningChoice:
        """Return the lowest predicted finish-time value without random exploration."""

        candidates = engine_candidates(
            minimum=self._settings.min_raw_value,
            maximum=self._settings.max_raw_value,
        )
        projection = self._context_projection(context, candidates)
        if projection.context_finish_count < self._shape.sampling_warmup_successes:
            engine_raw = _midpoint_candidate(candidates)
            return self._choice_for(
                context,
                engine_raw,
                estimate=projection.estimates[engine_raw],
                sampled_score=None,
            )
        best: EngineTuningChoice | None = None
        for engine_raw in candidates:
            estimate = projection.estimates[engine_raw]
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

        _ = seed, draws
        candidates = engine_candidates(
            minimum=self._settings.min_raw_value,
            maximum=self._settings.max_raw_value,
        )
        if not candidates:
            raise ValueError("adaptive engine tuning has no engine candidates")
        projection = self._context_projection(context, candidates)
        probabilities = _sampling_probabilities(
            projection=projection,
            candidates=candidates,
            settings=self._settings,
            shape=self._shape,
        )
        return tuple(
            EngineTuningCandidateEstimate(
                engine_setting_raw_value=engine_raw,
                probability=probabilities[engine_raw],
                mean_score=projection.estimates[engine_raw].mean_score,
                uncertainty_score=projection.estimates[engine_raw].uncertainty_score,
                estimated_finish_time_ms=finish_time_ms_from_score(
                    projection.estimates[engine_raw].mean_score
                ),
                finish_count=projection.estimates[engine_raw].exact_finish_count,
                best_finish_time_ms=projection.estimates[engine_raw].best_finish_time_ms,
            )
            for engine_raw in candidates
        )

    def record(self, outcome: EngineTuningEpisodeOutcome) -> EngineTuningRuntimeState:
        """Update the state from one terminal episode result."""

        return self.record_many((outcome,))

    def record_many(
        self,
        outcomes: tuple[EngineTuningEpisodeOutcome, ...],
    ) -> EngineTuningRuntimeState:
        """Train the ensemble from this successful rollout batch."""

        successful = tuple(_successful_score(outcome) for outcome in outcomes)
        successful = tuple(item for item in successful if item is not None)
        if not successful:
            return self._state
        self._state = self._state.with_model_state(
            _fit_model_state(
                successful,
                settings=self._settings,
                shape=self._shape,
                previous=self._state.model_state,
                update_count=self._state.update_count,
            ),
            increment_update_count=True,
        )
        return self._state

    def score(self, outcome: EngineTuningEpisodeOutcome) -> float:
        """Return a higher-is-better negative finish-time score."""

        finish_time_ms = successful_finish_time_ms(outcome)
        if finish_time_ms is None:
            return self._prior_score()
        return finish_time_score(finish_time_ms)

    def _best_choice_from_scores(
        self,
        *,
        context: EngineTuningContext,
        candidates: tuple[int, ...],
        projection: _EngineProjection,
        member_index: int,
    ) -> EngineTuningChoice:
        best: EngineTuningChoice | None = None
        for engine_raw in candidates:
            estimate = projection.estimates[engine_raw]
            scores = projection.member_scores.get(engine_raw, ())
            sampled_score = (
                estimate.mean_score
                if not scores
                else scores[min(member_index, len(scores) - 1)]
            )
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

    def _choice_for(
        self,
        context: EngineTuningContext,
        engine_raw: int,
        *,
        estimate: _EngineEstimate,
        sampled_score: float | None,
    ) -> EngineTuningChoice:
        return EngineTuningChoice(
            context=context,
            engine_setting_raw_value=engine_raw,
            sampled_score=estimate.mean_score if sampled_score is None else sampled_score,
            mean_score=estimate.mean_score,
            finish_count=estimate.exact_finish_count,
            estimated_finish_time_ms=finish_time_ms_from_score(estimate.mean_score),
            best_finish_time_ms=estimate.best_finish_time_ms,
        )

    def _context_projection(
        self,
        context: EngineTuningContext,
        candidates: tuple[int, ...],
    ) -> _EngineProjection:
        model_state = self._state.model_state
        estimates: dict[int, _EngineEstimate] = {}
        member_scores: dict[int, tuple[float, ...]] = {}
        if model_state is None or model_state.backend != "mlp_ensemble":
            for engine_raw in candidates:
                estimates[engine_raw] = _EngineEstimate(
                    mean_score=self._prior_score(),
                    uncertainty_score=0.0,
                    exact_finish_count=0,
                    best_finish_time_ms=None,
                )
            return _EngineProjection(
                estimates=estimates,
                member_scores={},
                member_count=self._shape.member_count,
                context_finish_count=0,
            )

        context_finish_count = _context_finish_count(model_state, context)
        raw_member_scores = _predict_member_scores(
            model_state,
            context=context,
            candidates=candidates,
            settings=self._settings,
            shape=self._shape,
        )
        for engine_raw in candidates:
            scores = raw_member_scores.get(engine_raw, ())
            if not scores:
                mean_score = self._prior_score()
                uncertainty_score = 0.0
            else:
                mean_score = sum(scores) / len(scores)
                uncertainty_score = _std(scores)
            estimates[engine_raw] = _EngineEstimate(
                mean_score=mean_score,
                uncertainty_score=uncertainty_score,
                exact_finish_count=context_finish_count,
                best_finish_time_ms=None,
            )
            member_scores[engine_raw] = scores
        return _EngineProjection(
            estimates=estimates,
            member_scores=member_scores,
            member_count=max(1, len(model_state.members)),
            context_finish_count=context_finish_count,
        )

    def _prior_score(self) -> float:
        return -max(1.0, float(self._settings.prior_finish_time_seconds))


@dataclass(frozen=True, slots=True)
class _EngineEstimate:
    mean_score: float
    uncertainty_score: float
    exact_finish_count: int
    best_finish_time_ms: int | None


@dataclass(frozen=True, slots=True)
class _EngineProjection:
    estimates: dict[int, _EngineEstimate]
    member_scores: dict[int, tuple[float, ...]]
    member_count: int
    context_finish_count: int


class _EngineTuningMember(torch.nn.Module):
    def __init__(
        self,
        *,
        course_count: int,
        vehicle_count: int,
        shape: EngineTuningEnsembleShape,
        prior_score: float,
        prior_seed: int,
        randomized_prior_seconds: float,
    ) -> None:
        super().__init__()
        self.shape = shape
        self.prior_score = float(prior_score)
        self.randomized_prior_seconds = max(0.0, float(randomized_prior_seconds))
        self.course_embedding = torch.nn.Embedding(course_count, shape.course_embedding_dim)
        self.vehicle_embedding = torch.nn.Embedding(vehicle_count, shape.vehicle_embedding_dim)
        input_dim = (
            shape.course_embedding_dim
            + shape.vehicle_embedding_dim
            + shape.engine_basis_count
        )
        self.residual_net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, shape.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(shape.hidden_dim, shape.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(shape.hidden_dim, 1),
        )
        output = self.residual_net[-1]
        if isinstance(output, torch.nn.Linear):
            torch.nn.init.zeros_(output.weight)
            torch.nn.init.zeros_(output.bias)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(prior_seed))
            self.random_prior_net = torch.nn.Sequential(
                torch.nn.Linear(input_dim, shape.hidden_dim),
                torch.nn.Tanh(),
                torch.nn.Linear(shape.hidden_dim, 1),
            )
        for parameter in self.random_prior_net.parameters():
            parameter.requires_grad_(False)

    def forward(
        self,
        course_indices: torch.Tensor,
        vehicle_indices: torch.Tensor,
        engine_values: torch.Tensor,
    ) -> torch.Tensor:
        features = self._features(course_indices, vehicle_indices, engine_values)
        residual = self.residual_net(features).squeeze(-1)
        prior = self.random_prior_net(features).squeeze(-1)
        return self.prior_score + residual + self.randomized_prior_seconds * prior

    def _features(
        self,
        course_indices: torch.Tensor,
        vehicle_indices: torch.Tensor,
        engine_values: torch.Tensor,
    ) -> torch.Tensor:
        basis = _engine_basis(engine_values, self.shape)
        features = torch.cat(
            (
                self.course_embedding(course_indices),
                self.vehicle_embedding(vehicle_indices),
                basis,
            ),
            dim=-1,
        )
        return features


def _fit_model_state(
    successful: tuple[_SuccessfulOutcome, ...],
    *,
    settings: MlpEnsembleEngineTunerSettings,
    shape: EngineTuningEnsembleShape,
    previous: EngineTuningModelState | None,
    update_count: int,
) -> EngineTuningModelState | None:
    if not successful:
        return previous if previous is not None and previous.backend == "mlp_ensemble" else None
    previous_state = (
        previous if previous is not None and previous.backend == "mlp_ensemble" else None
    )
    course_keys = _ordered_vocab(
        previous_state.course_keys if previous_state else (),
        (outcome.context.course_key for outcome, _, _ in successful),
    )
    vehicle_ids = _ordered_vocab(
        previous_state.vehicle_ids if previous_state else (),
        (outcome.context.vehicle_id for outcome, _, _ in successful),
    )
    course_index = {key: index for index, key in enumerate(course_keys)}
    vehicle_index = {key: index for index, key in enumerate(vehicle_ids)}
    train_data = _training_tensors(
        successful,
        course_index=course_index,
        vehicle_index=vehicle_index,
    )
    members: list[EngineTuningEnsembleMemberState] = []
    previous_members = previous_state.members if previous_state is not None else ()
    for member_index in range(shape.member_count):
        seed = update_count * 997 + _member_seed(member_index)
        model = _new_member(
            course_count=len(course_keys),
            vehicle_count=len(vehicle_ids),
            shape=shape,
            prior_score=-max(1.0, float(settings.prior_finish_time_seconds)),
            randomized_prior_seconds=settings.randomized_prior_seconds,
            seed=seed,
            prior_seed=_member_seed(member_index),
        )
        if member_index < len(previous_members) and previous_state is not None:
            _load_member_state(
                model,
                previous_members[member_index],
                previous_course_keys=previous_state.course_keys,
                current_course_keys=course_keys,
                previous_vehicle_ids=previous_state.vehicle_ids,
                current_vehicle_ids=vehicle_ids,
            )
        _train_member(
            model,
            train_data,
            shape=shape,
            seed=seed,
        )
        members.append(_member_state(model))
    return EngineTuningModelState(
        backend="mlp_ensemble",
        course_keys=course_keys,
        vehicle_ids=vehicle_ids,
        members=tuple(members),
        contexts=_updated_model_contexts(
            previous_state.contexts if previous_state else (),
            successful,
        ),
    )


@dataclass(frozen=True, slots=True)
class _TrainingTensors:
    course_indices: torch.Tensor
    vehicle_indices: torch.Tensor
    engine_values: torch.Tensor
    targets: torch.Tensor
    weights: torch.Tensor


def _training_tensors(
    successful: tuple[_SuccessfulOutcome, ...],
    *,
    course_index: dict[str, int],
    vehicle_index: dict[str, int],
) -> _TrainingTensors:
    return _TrainingTensors(
        course_indices=torch.as_tensor(
            [course_index[outcome.context.course_key] for outcome, _, _ in successful],
            dtype=torch.long,
        ),
        vehicle_indices=torch.as_tensor(
            [vehicle_index[outcome.context.vehicle_id] for outcome, _, _ in successful],
            dtype=torch.long,
        ),
        engine_values=torch.as_tensor(
            [outcome.engine_setting_raw_value / 100.0 for outcome, _, _ in successful],
            dtype=torch.float32,
        ),
        targets=torch.as_tensor([score for _, score, _ in successful], dtype=torch.float32),
        weights=torch.ones((len(successful),), dtype=torch.float32),
    )


def _new_member(
    *,
    course_count: int,
    vehicle_count: int,
    shape: EngineTuningEnsembleShape,
    prior_score: float,
    randomized_prior_seconds: float,
    seed: int,
    prior_seed: int,
) -> _EngineTuningMember:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        return _EngineTuningMember(
            course_count=course_count,
            vehicle_count=vehicle_count,
            shape=shape,
            prior_score=prior_score,
            prior_seed=prior_seed,
            randomized_prior_seconds=randomized_prior_seconds,
        )


def _train_member(
    model: _EngineTuningMember,
    train_data: _TrainingTensors,
    *,
    shape: EngineTuningEnsembleShape,
    seed: int,
) -> None:
    model.train()
    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=shape.learning_rate,
    )
    generator = torch.Generator().manual_seed(int(seed))
    for _ in range(shape.training_steps):
        optimizer.zero_grad()
        predictions = model(
            train_data.course_indices,
            train_data.vehicle_indices,
            train_data.engine_values,
        )
        bootstrap = (
            torch.rand(train_data.weights.shape, generator=generator)
            < shape.bootstrap_keep_probability
        ).to(dtype=torch.float32)
        if float(bootstrap.sum()) <= 0.0:
            bootstrap = torch.ones_like(train_data.weights)
        weights = train_data.weights * bootstrap
        loss = (weights * (predictions - train_data.targets).pow(2)).sum() / weights.sum()
        loss.backward()
        optimizer.step()
    model.eval()


def _predict_member_scores(
    state: EngineTuningModelState,
    *,
    context: EngineTuningContext,
    candidates: tuple[int, ...],
    settings: MlpEnsembleEngineTunerSettings,
    shape: EngineTuningEnsembleShape,
) -> dict[int, tuple[float, ...]]:
    if context.course_key not in state.course_keys or context.vehicle_id not in state.vehicle_ids:
        return {}
    course_index = state.course_keys.index(context.course_key)
    vehicle_index = state.vehicle_ids.index(context.vehicle_id)
    course_indices = torch.full((len(candidates),), course_index, dtype=torch.long)
    vehicle_indices = torch.full((len(candidates),), vehicle_index, dtype=torch.long)
    engine_values = torch.as_tensor(
        [candidate / 100.0 for candidate in candidates],
        dtype=torch.float32,
    )
    scores_by_engine = {candidate: [] for candidate in candidates}
    for member_index, member_state in enumerate(state.members):
        model = _EngineTuningMember(
            course_count=len(state.course_keys),
            vehicle_count=len(state.vehicle_ids),
            shape=shape,
            prior_score=-max(1.0, float(settings.prior_finish_time_seconds)),
            prior_seed=_member_seed(member_index),
            randomized_prior_seconds=settings.randomized_prior_seconds,
        )
        _load_member_state(
            model,
            member_state,
            previous_course_keys=state.course_keys,
            current_course_keys=state.course_keys,
            previous_vehicle_ids=state.vehicle_ids,
            current_vehicle_ids=state.vehicle_ids,
        )
        model.eval()
        with torch.no_grad():
            predictions = model(course_indices, vehicle_indices, engine_values)
        for engine_raw, score in zip(candidates, predictions.tolist(), strict=True):
            scores_by_engine[engine_raw].append(float(score))
    return {key: tuple(values) for key, values in scores_by_engine.items()}


def _member_state(model: _EngineTuningMember) -> EngineTuningEnsembleMemberState:
    tensors: list[EngineTuningTensorState] = []
    for name, tensor in model.state_dict().items():
        cpu_tensor = tensor.detach().to(dtype=torch.float32).cpu()
        tensors.append(
            EngineTuningTensorState(
                name=name,
                value=cpu_tensor,
            )
        )
    return EngineTuningEnsembleMemberState(tensors=tuple(tensors))


def _load_member_state(
    model: _EngineTuningMember,
    member_state: EngineTuningEnsembleMemberState,
    *,
    previous_course_keys: tuple[str, ...],
    current_course_keys: tuple[str, ...],
    previous_vehicle_ids: tuple[str, ...],
    current_vehicle_ids: tuple[str, ...],
) -> None:
    state_dict = model.state_dict()
    for tensor_state in member_state.tensors:
        if tensor_state.name not in state_dict:
            continue
        tensor = tensor_state.value.to(dtype=state_dict[tensor_state.name].dtype)
        if tensor_state.name == "course_embedding.weight":
            _copy_embedding_rows(
                state_dict[tensor_state.name],
                tensor,
                previous_keys=previous_course_keys,
                current_keys=current_course_keys,
            )
        elif tensor_state.name == "vehicle_embedding.weight":
            _copy_embedding_rows(
                state_dict[tensor_state.name],
                tensor,
                previous_keys=previous_vehicle_ids,
                current_keys=current_vehicle_ids,
            )
        elif tuple(state_dict[tensor_state.name].shape) == tuple(tensor.shape):
            state_dict[tensor_state.name].copy_(tensor)
    model.load_state_dict(state_dict)


def _copy_embedding_rows(
    target: torch.Tensor,
    source: torch.Tensor,
    *,
    previous_keys: tuple[str, ...],
    current_keys: tuple[str, ...],
) -> None:
    previous_index = {key: index for index, key in enumerate(previous_keys)}
    for current_index, key in enumerate(current_keys):
        source_index = previous_index.get(key)
        if source_index is None or source_index >= source.shape[0]:
            continue
        target[current_index].copy_(source[source_index])


def _sampling_probabilities(
    *,
    projection: _EngineProjection,
    candidates: tuple[int, ...],
    settings: MlpEnsembleEngineTunerSettings,
    shape: EngineTuningEnsembleShape,
) -> dict[int, float]:
    if not candidates:
        return {}
    uniform_probability = 1.0 / len(candidates)
    if (
        not projection.member_scores
        or projection.context_finish_count < shape.sampling_warmup_successes
    ):
        return {candidate: uniform_probability for candidate in candidates}

    temperature = max(
        shape.sampling_min_temperature_seconds,
        float(settings.randomized_prior_seconds) / sqrt(max(1, projection.context_finish_count)),
    )
    acquisition_scores = tuple(
        projection.estimates[candidate].mean_score
        + projection.estimates[candidate].uncertainty_score
        for candidate in candidates
    )
    model_probabilities = _softmax_probabilities(acquisition_scores, temperature=temperature)
    warmup = max(1.0, float(shape.sampling_warmup_successes))
    model_weight = projection.context_finish_count / (projection.context_finish_count + warmup)
    exploration = max(0.0, min(1.0, float(settings.uniform_exploration)))
    return {
        candidate: exploration * uniform_probability
        + (1.0 - exploration)
        * (
            (1.0 - model_weight) * uniform_probability
            + model_weight * model_probabilities[index]
        )
        for index, candidate in enumerate(candidates)
    }


def _softmax_probabilities(scores: tuple[float, ...], *, temperature: float) -> tuple[float, ...]:
    if not scores:
        return ()
    clamped_temperature = max(1.0e-6, float(temperature))
    maximum = max(scores)
    weights = tuple(exp((score - maximum) / clamped_temperature) for score in scores)
    total = sum(weights)
    if total <= 0.0:
        return tuple(1.0 / len(scores) for _ in scores)
    return tuple(weight / total for weight in weights)


def _sample_candidate(
    candidates: tuple[int, ...],
    *,
    probabilities: dict[int, float],
    rng: Random,
) -> int:
    threshold = rng.random()
    cumulative = 0.0
    for candidate in candidates:
        cumulative += max(0.0, probabilities.get(candidate, 0.0))
        if threshold <= cumulative:
            return candidate
    return candidates[-1]


def _midpoint_candidate(candidates: tuple[int, ...]) -> int:
    if not candidates:
        raise ValueError("adaptive engine tuning has no engine candidates")
    midpoint = (candidates[0] + candidates[-1]) / 2.0
    return min(candidates, key=lambda candidate: (abs(candidate - midpoint), candidate))


def _context_finish_count(
    state: EngineTuningModelState,
    context: EngineTuningContext,
) -> int:
    for item in state.contexts:
        if item.context_key == context.key:
            return max(0, item.finish_count)
    return 0


def _successful_score(
    outcome: EngineTuningEpisodeOutcome,
) -> tuple[EngineTuningEpisodeOutcome, float, int] | None:
    finish_time_ms = successful_finish_time_ms(outcome)
    if finish_time_ms is None:
        return None
    return outcome, finish_time_score(finish_time_ms), finish_time_ms


def _mlp_state_or_empty(state: EngineTuningRuntimeState | None) -> EngineTuningRuntimeState:
    if state is None or state.model_state is None or state.model_state.backend != "mlp_ensemble":
        return empty_engine_tuning_state()
    return EngineTuningRuntimeState(
        version=state.version,
        update_count=state.update_count,
        candidates=(),
        model_state=state.model_state,
    )


def _updated_model_contexts(
    previous: tuple[EngineTuningModelContextState, ...],
    successful: tuple[_SuccessfulOutcome, ...],
) -> tuple[EngineTuningModelContextState, ...]:
    contexts = {context.context_key: context for context in previous}
    for outcome, _, _ in successful:
        existing = contexts.get(outcome.context.key)
        contexts[outcome.context.key] = EngineTuningModelContextState(
            context_key=outcome.context.key,
            course_key=outcome.context.course_key,
            vehicle_id=outcome.context.vehicle_id,
            finish_count=1 if existing is None else existing.finish_count + 1,
        )
    return tuple(sorted(contexts.values(), key=lambda item: item.context_key))


def _ordered_vocab(previous: tuple[str, ...], observed: Iterable[str]) -> tuple[str, ...]:
    values = list(previous)
    for value in sorted(set(item for item in observed if isinstance(item, str))):
        if value not in values:
            values.append(value)
    return tuple(values)


def _std(values: tuple[float, ...]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = sum(values) / len(values)
    return sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _engine_basis(engine_values: torch.Tensor, shape: EngineTuningEnsembleShape) -> torch.Tensor:
    centers = torch.linspace(
        0.0,
        1.0,
        max(2, int(shape.engine_basis_count)),
        dtype=engine_values.dtype,
        device=engine_values.device,
    )
    width = max(0.001, float(shape.engine_basis_width_raw) / 100.0)
    distances = (engine_values.unsqueeze(-1) - centers) / width
    basis = torch.exp(-0.5 * distances.pow(2))
    return basis / basis.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)


def _member_seed(member_index: int) -> int:
    return 104_729 + int(member_index) * 7_919


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
