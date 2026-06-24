# src/rl_fzerox/core/engine_tuning/experimental/greedy.py
"""Greedy selection helpers for experimental engine-tuning backends."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import exp

from rl_fzerox.core.engine_tuning.types import (
    ENGINE_TUNER_DEFAULTS,
    EngineTuningCandidateEstimate,
)


@dataclass(frozen=True, slots=True)
class StableGreedySelection:
    """Risk-adjusted plateau rule for deterministic engine choices."""

    uncertainty_penalty: float = 0.5
    plateau_tolerance_seconds: float = ENGINE_TUNER_DEFAULTS.greedy_plateau_tolerance_seconds
    boundary_softness_fraction: float = 0.25
    minimum_boundary_softness_seconds: float = 0.05


STABLE_GREEDY_SELECTION = StableGreedySelection()


def stable_greedy_engine_setting(
    estimates: Sequence[EngineTuningCandidateEstimate],
    *,
    selection: StableGreedySelection | None = None,
) -> int | None:
    """Return a soft plateau center for deterministic engine choices."""

    if selection is None:
        selection = STABLE_GREEDY_SELECTION
    ordered = tuple(sorted(estimates, key=lambda estimate: estimate.engine_setting_raw_value))
    if not ordered:
        return None
    scored = tuple(
        (estimate, _risk_adjusted_score(estimate, selection=selection)) for estimate in ordered
    )
    best_score = max(score for _, score in scored)
    weighted_total = 0.0
    weight_total = 0.0
    for estimate, score in scored:
        seconds_slower = best_score - score
        weight = _soft_plateau_weight(seconds_slower, selection=selection)
        weighted_total += estimate.engine_setting_raw_value * weight
        weight_total += weight
    if weight_total <= 0.0:
        return max(scored, key=lambda item: item[1])[0].engine_setting_raw_value
    target = weighted_total / weight_total
    return min(
        ordered,
        key=lambda estimate: (
            abs(estimate.engine_setting_raw_value - target),
            estimate.engine_setting_raw_value,
        ),
    ).engine_setting_raw_value


def _risk_adjusted_score(
    estimate: EngineTuningCandidateEstimate,
    *,
    selection: StableGreedySelection,
) -> float:
    return estimate.mean_score - selection.uncertainty_penalty * max(
        0.0,
        estimate.uncertainty_score,
    )


def _soft_plateau_weight(
    seconds_slower: float,
    *,
    selection: StableGreedySelection,
) -> float:
    boundary_softness = max(
        selection.minimum_boundary_softness_seconds,
        selection.plateau_tolerance_seconds * selection.boundary_softness_fraction,
    )
    exponent = (seconds_slower - selection.plateau_tolerance_seconds) / boundary_softness
    if exponent > 60.0:
        return 0.0
    if exponent < -60.0:
        return 1.0
    return 1.0 / (1.0 + exp(exponent))
