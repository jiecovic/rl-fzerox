# src/rl_fzerox/ui/watch/view/auxiliary_metrics.py
from __future__ import annotations

from dataclasses import dataclass, field
from math import log
from typing import Protocol, TypedDict

from rl_fzerox.core.policy.auxiliary_state.heads import AuxiliaryStateLossTerm
from rl_fzerox.core.policy.auxiliary_state.targets import (
    AuxiliaryStateTargetName,
    auxiliary_state_target_bounds,
    resolve_auxiliary_state_target,
)
from rl_fzerox.core.runtime_spec.schema import PolicyConfig


class _AuxiliaryMetricsSnapshot(Protocol):
    @property
    def episode(self) -> int: ...

    @property
    def policy_decision_frame(self) -> bool: ...

    @property
    def policy_auxiliary_state_predictions(self) -> dict[str, object] | None: ...

    @property
    def policy_auxiliary_state_targets(self) -> dict[str, object] | None: ...


class _CategoricalPrediction(TypedDict):
    index: int
    probabilities: list[float]
    confidence: float


_EPSILON = 1e-6


@dataclass(frozen=True, slots=True)
class AuxiliaryEpisodeMetric:
    name: AuxiliaryStateTargetName
    sample_count: int = 0
    mean_loss: float = 0.0
    mean_error_percent: float | None = None
    accuracy: float | None = None
    mean_confidence: float | None = None


@dataclass(frozen=True, slots=True)
class AuxiliaryEpisodeMetricsSnapshot:
    episode: int
    metrics: tuple[AuxiliaryEpisodeMetric, ...]


@dataclass(slots=True)
class _AuxiliaryRunningMetric:
    sample_count: int = 0
    loss_sum: float = 0.0
    error_percent_sum: float = 0.0
    error_percent_count: int = 0
    correct_count: int = 0
    accuracy_count: int = 0
    confidence_sum: float = 0.0
    confidence_count: int = 0

    def add_scalar(self, *, loss: float, error_percent: float) -> None:
        self.sample_count += 1
        self.loss_sum += loss
        self.error_percent_sum += error_percent
        self.error_percent_count += 1

    def add_binary(self, *, loss: float, correct: bool) -> None:
        self.sample_count += 1
        self.loss_sum += loss
        self.correct_count += int(correct)
        self.accuracy_count += 1

    def add_categorical(
        self,
        *,
        loss: float,
        correct: bool,
        confidence: float,
    ) -> None:
        self.sample_count += 1
        self.loss_sum += loss
        self.correct_count += int(correct)
        self.accuracy_count += 1
        self.confidence_sum += confidence
        self.confidence_count += 1


@dataclass(slots=True)
class AuxiliaryEpisodeMetricsTracker:
    loss_terms: tuple[AuxiliaryStateLossTerm, ...]
    episode: int | None = None
    metrics: dict[AuxiliaryStateTargetName, _AuxiliaryRunningMetric] = field(default_factory=dict)

    @classmethod
    def from_policy_config(
        cls,
        policy_config: PolicyConfig | None,
    ) -> AuxiliaryEpisodeMetricsTracker:
        if policy_config is None or not policy_config.auxiliary_state.enabled:
            return cls(loss_terms=())
        terms = tuple(
            AuxiliaryStateLossTerm(
                name=loss.name,
                weight=float(loss.weight),
                grounded_only=bool(loss.grounded_only),
            )
            for loss in policy_config.auxiliary_state.losses
        )
        return cls(loss_terms=terms)

    def observe_snapshot(self, snapshot: _AuxiliaryMetricsSnapshot) -> None:
        if not self.loss_terms or not snapshot.policy_decision_frame:
            return
        if self.episode != snapshot.episode:
            self.episode = snapshot.episode
            self.metrics = {term.name: _AuxiliaryRunningMetric() for term in self.loss_terms}
        predictions = snapshot.policy_auxiliary_state_predictions or {}
        targets = snapshot.policy_auxiliary_state_targets or {}
        airborne_target = targets.get("vehicle_state.airborne")
        airborne = bool(isinstance(airborne_target, float | int) and float(airborne_target) >= 0.5)
        for loss_term in self.loss_terms:
            if loss_term.grounded_only and airborne:
                continue
            prediction = predictions.get(loss_term.name)
            target = targets.get(loss_term.name)
            if prediction is None or target is None:
                continue
            self._observe_one(loss_term.name, prediction=prediction, target=target)

    def snapshot(self) -> AuxiliaryEpisodeMetricsSnapshot | None:
        if self.episode is None or not self.loss_terms:
            return None
        metrics: list[AuxiliaryEpisodeMetric] = []
        for loss_term in self.loss_terms:
            aggregate = self.metrics.get(loss_term.name, _AuxiliaryRunningMetric())
            count = aggregate.sample_count
            metrics.append(
                AuxiliaryEpisodeMetric(
                    name=loss_term.name,
                    sample_count=count,
                    mean_loss=0.0 if count == 0 else aggregate.loss_sum / count,
                    mean_error_percent=(
                        None
                        if aggregate.error_percent_count == 0
                        else aggregate.error_percent_sum / aggregate.error_percent_count
                    ),
                    accuracy=(
                        None
                        if aggregate.accuracy_count == 0
                        else aggregate.correct_count / aggregate.accuracy_count
                    ),
                    mean_confidence=(
                        None
                        if aggregate.confidence_count == 0
                        else aggregate.confidence_sum / aggregate.confidence_count
                    ),
                )
            )
        return AuxiliaryEpisodeMetricsSnapshot(
            episode=self.episode,
            metrics=tuple(metrics),
        )

    def _observe_one(
        self,
        name: AuxiliaryStateTargetName,
        *,
        prediction: object,
        target: object,
    ) -> None:
        aggregate = self.metrics.setdefault(name, _AuxiliaryRunningMetric())
        target_spec = resolve_auxiliary_state_target(name)
        if target_spec.kind == "scalar":
            predicted = _scalar_value(prediction)
            target_value = _scalar_value(target)
            if predicted is None or target_value is None:
                return
            aggregate.add_scalar(
                loss=_smooth_l1(predicted - target_value),
                error_percent=_error_percent(name, predicted=predicted, target=target_value),
            )
            return
        if target_spec.kind == "binary":
            predicted_probability = _scalar_value(prediction)
            target_value = _scalar_value(target)
            if predicted_probability is None or target_value is None:
                return
            aggregate.add_binary(
                loss=_binary_cross_entropy(predicted_probability, target_value),
                correct=(predicted_probability >= 0.5) == (target_value >= 0.5),
            )
            return
        categorical_prediction = _categorical_prediction(prediction)
        categorical_target = _categorical_target(target)
        if categorical_prediction is None or categorical_target is None:
            return
        aggregate.add_categorical(
            loss=_categorical_cross_entropy(categorical_prediction, categorical_target),
            correct=categorical_prediction["index"] == categorical_target,
            confidence=categorical_prediction["confidence"],
        )


def _scalar_value(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _smooth_l1(delta: float) -> float:
    absolute = abs(delta)
    if absolute < 1.0:
        return 0.5 * absolute * absolute
    return absolute - 0.5


def _error_percent(
    name: AuxiliaryStateTargetName,
    *,
    predicted: float,
    target: float,
) -> float:
    low, high = auxiliary_state_target_bounds(name)
    span = max(high - low, _EPSILON)
    return abs(predicted - target) / span * 100.0


def _binary_cross_entropy(predicted_probability: float, target: float) -> float:
    probability = min(max(predicted_probability, _EPSILON), 1.0 - _EPSILON)
    target_value = 1.0 if target >= 0.5 else 0.0
    return -(target_value * log(probability) + (1.0 - target_value) * log(1.0 - probability))


def _categorical_prediction(
    value: object,
) -> _CategoricalPrediction | None:
    if not isinstance(value, dict):
        return None
    raw_index = value.get("index")
    raw_probabilities = value.get("probabilities")
    if not isinstance(raw_index, int) or not isinstance(raw_probabilities, list):
        return None
    probabilities = [
        float(probability)
        for probability in raw_probabilities
        if isinstance(probability, int | float)
    ]
    if len(probabilities) != len(raw_probabilities):
        return None
    if not (0 <= raw_index < len(probabilities)):
        return None
    raw_confidence = value.get("confidence")
    confidence = (
        float(raw_confidence)
        if isinstance(raw_confidence, int | float)
        else float(probabilities[raw_index])
    )
    return {
        "index": raw_index,
        "probabilities": probabilities,
        "confidence": confidence,
    }


def _categorical_target(value: object) -> int | None:
    if not isinstance(value, dict):
        return None
    raw_index = value.get("index")
    return raw_index if isinstance(raw_index, int) else None


def _categorical_cross_entropy(
    prediction: _CategoricalPrediction,
    target_index: int,
) -> float:
    probabilities = prediction["probabilities"]
    if not isinstance(probabilities, list) or not (0 <= target_index < len(probabilities)):
        return 0.0
    target_probability = min(
        max(float(probabilities[target_index]), _EPSILON),
        1.0,
    )
    return -log(target_probability)
