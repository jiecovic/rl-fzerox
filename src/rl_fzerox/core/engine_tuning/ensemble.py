# src/rl_fzerox/core/engine_tuning/ensemble.py
"""Bootstrapped MLP ensemble backend for adaptive engine tuning."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import sqrt
from random import Random

import torch

from rl_fzerox.core.engine_tuning.state import (
    EngineTuningCandidateState,
    EngineTuningEnsembleMemberState,
    EngineTuningModelState,
    EngineTuningRuntimeState,
    EngineTuningTensorState,
    empty_engine_tuning_state,
)
from rl_fzerox.core.engine_tuning.types import (
    EngineTunerSettings,
    EngineTuningCandidateEstimate,
    EngineTuningChoice,
    EngineTuningContext,
    EngineTuningEpisodeOutcome,
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
    hidden_dim: int = 32
    training_steps: int = 48
    learning_rate: float = 0.004
    bootstrap_keep_probability: float = 0.8


ENGINE_TUNING_ENSEMBLE_SHAPE = EngineTuningEnsembleShape()
_SuccessfulOutcome = tuple[EngineTuningEpisodeOutcome, float, int]


class MlpEnsembleEngineTuner:
    """Choose engine values from a bootstrapped neural finish-time surrogate."""

    def __init__(
        self,
        *,
        settings: EngineTunerSettings,
        state: EngineTuningRuntimeState | None = None,
    ) -> None:
        self._settings = settings
        self._state = _mlp_state_or_empty(state)
        self._shape = ENGINE_TUNING_ENSEMBLE_SHAPE

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
        if rng.random() < max(0.0, min(1.0, self._settings.uniform_exploration)):
            selected = rng.choice(candidates)
            return self._choice_for(
                context,
                selected,
                estimate=projection.estimates[selected],
                sampled_score=None,
            )
        member_index = rng.randrange(max(1, projection.member_count))
        return self._best_choice_from_scores(
            context=context,
            candidates=candidates,
            projection=projection,
            member_index=member_index,
        )

    def recommendation(self, context: EngineTuningContext) -> EngineTuningChoice:
        """Return the lowest predicted finish-time value without random exploration."""

        candidates = engine_candidates(
            minimum=self._settings.min_raw_value,
            maximum=self._settings.max_raw_value,
        )
        projection = self._context_projection(context, candidates)
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
            if rng.random() < max(0.0, min(1.0, self._settings.uniform_exploration)):
                counts[rng.choice(candidates)] += 1
                continue
            member_index = rng.randrange(max(1, projection.member_count))
            choice = self._best_choice_from_scores(
                context=context,
                candidates=candidates,
                projection=projection,
                member_index=member_index,
            )
            counts[choice.engine_setting_raw_value] += 1
        return tuple(
            EngineTuningCandidateEstimate(
                engine_setting_raw_value=engine_raw,
                probability=counts[engine_raw] / draw_count,
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
        """Update diagnostics and train the ensemble from this successful batch."""

        successful = tuple(_successful_score(outcome) for outcome in outcomes)
        successful = tuple(item for item in successful if item is not None)
        if not successful:
            return self._state
        next_state = self._state.decay(self._settings.stat_decay)
        for outcome, score, finish_time_ms in successful:
            candidate = _candidate_from_state(
                next_state,
                outcome.context,
                outcome.engine_setting_raw_value,
            ).record(
                score=score,
                finish_time_ms=finish_time_ms,
            )
            next_state = next_state.with_candidate(candidate)
        self._state = next_state.with_model_state(
            _fit_model_state(
                successful,
                settings=self._settings,
                shape=self._shape,
                previous=next_state.model_state,
                update_count=next_state.update_count,
            )
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
        exact_candidate = self._state.candidate_map().get((context.key, int(engine_raw)))
        return EngineTuningChoice(
            context=context,
            engine_setting_raw_value=engine_raw,
            sampled_score=estimate.mean_score if sampled_score is None else sampled_score,
            mean_score=estimate.mean_score,
            finish_count=0 if exact_candidate is None else exact_candidate.finish_count,
            estimated_finish_time_ms=finish_time_ms_from_score(estimate.mean_score),
            best_finish_time_ms=None if exact_candidate is None else exact_candidate.best_time_ms,
        )

    def _context_projection(
        self,
        context: EngineTuningContext,
        candidates: tuple[int, ...],
    ) -> _EngineProjection:
        model_state = self._state.model_state
        candidate_map = self._state.candidate_map()
        estimates: dict[int, _EngineEstimate] = {}
        member_scores: dict[int, tuple[float, ...]] = {}
        if model_state is None or model_state.backend != "mlp_ensemble":
            for engine_raw in candidates:
                candidate = candidate_map.get((context.key, int(engine_raw)))
                estimates[engine_raw] = _EngineEstimate(
                    mean_score=self._prior_score(),
                    uncertainty_score=max(0.0, float(self._settings.exploration_seconds)),
                    exact_finish_count=0 if candidate is None else candidate.finish_count,
                    best_finish_time_ms=None if candidate is None else candidate.best_time_ms,
                )
            return _EngineProjection(
                estimates=estimates,
                member_scores={},
                member_count=self._shape.member_count,
            )

        raw_member_scores = _predict_member_scores(
            model_state,
            context=context,
            candidates=candidates,
            settings=self._settings,
            shape=self._shape,
        )
        for engine_raw in candidates:
            candidate = candidate_map.get((context.key, int(engine_raw)))
            scores = raw_member_scores.get(engine_raw, ())
            if not scores:
                mean_score = self._prior_score()
                uncertainty_score = max(0.0, float(self._settings.exploration_seconds))
            else:
                mean_score = sum(scores) / len(scores)
                uncertainty_score = _std(scores)
            estimates[engine_raw] = _EngineEstimate(
                mean_score=mean_score,
                uncertainty_score=uncertainty_score,
                exact_finish_count=0 if candidate is None else candidate.finish_count,
                best_finish_time_ms=None if candidate is None else candidate.best_time_ms,
            )
            member_scores[engine_raw] = scores
        return _EngineProjection(
            estimates=estimates,
            member_scores=member_scores,
            member_count=max(1, len(model_state.members)),
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


class _EngineTuningMember(torch.nn.Module):
    def __init__(
        self,
        *,
        course_count: int,
        vehicle_count: int,
        shape: EngineTuningEnsembleShape,
        prior_score: float,
    ) -> None:
        super().__init__()
        self.course_embedding = torch.nn.Embedding(course_count, shape.course_embedding_dim)
        self.vehicle_embedding = torch.nn.Embedding(vehicle_count, shape.vehicle_embedding_dim)
        input_dim = shape.course_embedding_dim + shape.vehicle_embedding_dim + 1
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, shape.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(shape.hidden_dim, shape.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(shape.hidden_dim, 1),
        )
        output = self.net[-1]
        if isinstance(output, torch.nn.Linear):
            torch.nn.init.zeros_(output.weight)
            torch.nn.init.constant_(output.bias, prior_score)

    def forward(
        self,
        course_indices: torch.Tensor,
        vehicle_indices: torch.Tensor,
        engine_values: torch.Tensor,
    ) -> torch.Tensor:
        features = torch.cat(
            (
                self.course_embedding(course_indices),
                self.vehicle_embedding(vehicle_indices),
                engine_values.unsqueeze(-1),
            ),
            dim=-1,
        )
        return self.net(features).squeeze(-1)


def _fit_model_state(
    successful: tuple[_SuccessfulOutcome, ...],
    *,
    settings: EngineTunerSettings,
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
        seed = update_count * 997 + member_index * 7919
        model = _new_member(
            course_count=len(course_keys),
            vehicle_count=len(vehicle_ids),
            shape=shape,
            prior_score=-max(1.0, float(settings.prior_finish_time_seconds)),
            seed=seed,
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
    seed: int,
) -> _EngineTuningMember:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        return _EngineTuningMember(
            course_count=course_count,
            vehicle_count=vehicle_count,
            shape=shape,
            prior_score=prior_score,
        )


def _train_member(
    model: _EngineTuningMember,
    train_data: _TrainingTensors,
    *,
    shape: EngineTuningEnsembleShape,
    seed: int,
) -> None:
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=shape.learning_rate)
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
    settings: EngineTunerSettings,
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
    for member_state in state.members:
        model = _EngineTuningMember(
            course_count=len(state.course_keys),
            vehicle_count=len(state.vehicle_ids),
            shape=shape,
            prior_score=-max(1.0, float(settings.prior_finish_time_seconds)),
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
                shape=tuple(int(dimension) for dimension in cpu_tensor.shape),
                values=tuple(float(value) for value in cpu_tensor.reshape(-1).tolist()),
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
        tensor = torch.as_tensor(tensor_state.values, dtype=state_dict[tensor_state.name].dtype)
        tensor = tensor.reshape(tensor_state.shape)
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
    return state


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
