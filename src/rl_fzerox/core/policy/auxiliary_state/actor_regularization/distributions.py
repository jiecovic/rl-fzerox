# src/rl_fzerox/core/policy/auxiliary_state/actor_regularization/distributions.py
"""Distribution readers for actor regularization over hybrid action spaces.

These helpers extract means, standard deviations, entropies, and lean/pitch
probabilities from SB3X hybrid distribution objects. Loss code receives plain
tensors instead of probing the dynamic distribution shape itself.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import torch


@dataclass(frozen=True, slots=True)
class _AxisDistributionStats:
    mean: torch.Tensor
    std: torch.Tensor
    entropy: torch.Tensor
    source: Literal["continuous", "discrete"]
    log_std: torch.Tensor | None = None


def _continuous_action_mode(distribution: object) -> torch.Tensor:
    continuous_dist = getattr(distribution, "continuous_dist", None)
    mode = getattr(continuous_dist, "mode", None)
    if not callable(mode):
        raise TypeError("Actor regularization requires a hybrid continuous distribution")
    value = mode()
    if not isinstance(value, torch.Tensor):
        raise TypeError("Continuous distribution mode must return a tensor")
    return value


def _continuous_action_log_std(distribution: object) -> torch.Tensor:
    continuous_log_std = getattr(distribution, "continuous_log_std", None)
    if not callable(continuous_log_std):
        raise TypeError("Actor regularization requires hybrid continuous log std access")
    value = continuous_log_std()
    if not isinstance(value, torch.Tensor):
        raise TypeError("Continuous distribution log std must return a tensor")
    log_std: torch.Tensor = value
    if log_std.ndim != 2:
        raise TypeError("Continuous distribution log std must be batched")
    return log_std


def _discrete_pitch_distribution_stats(
    distribution: object,
    *,
    discrete_pitch_index: int,
    bucket_values: tuple[float, ...],
) -> _AxisDistributionStats:
    branch_distribution = _discrete_branch_distribution(
        distribution,
        branch_index=discrete_pitch_index,
    )
    raw_probs = getattr(branch_distribution, "probs", None)
    if not isinstance(raw_probs, torch.Tensor):
        raise TypeError("Discrete pitch distribution must expose categorical probabilities")
    probs: torch.Tensor = raw_probs
    if probs.ndim != 2:
        raise TypeError("Discrete pitch probabilities must be batched")
    if probs.shape[1] != len(bucket_values):
        raise ValueError("Discrete pitch bucket count does not match the pitch distribution shape")

    bucket_tensor = probs.new_tensor(bucket_values).unsqueeze(0)
    mean = (probs * bucket_tensor).sum(dim=1)
    variance = (probs * (bucket_tensor - mean.unsqueeze(1)).square()).sum(dim=1)
    std = variance.clamp(min=0.0).sqrt()

    entropy_fn = getattr(branch_distribution, "entropy", None)
    if not callable(entropy_fn):
        raise TypeError("Discrete pitch distribution must expose entropy")
    entropy = entropy_fn()
    if not isinstance(entropy, torch.Tensor):
        raise TypeError("Discrete pitch entropy must be a tensor")
    return _AxisDistributionStats(mean=mean, std=std, entropy=entropy, source="discrete")


def _categorical_lean_expected_signed_values(
    distribution: object,
    *,
    branch_index: int,
) -> torch.Tensor:
    probs = _discrete_branch_probabilities(
        distribution,
        branch_index=branch_index,
        label="categorical lean",
    )
    if probs.shape[1] not in {3, 4}:
        raise ValueError("Categorical lean probabilities must have three or four values")
    signed_values = probs.new_zeros((probs.shape[1],))
    signed_values[1] = -1.0
    signed_values[2] = 1.0
    return (probs * signed_values.unsqueeze(0)).sum(dim=1)


def _split_lean_expected_signed_values(
    distribution: object,
    *,
    left_branch_index: int,
    right_branch_index: int,
) -> torch.Tensor:
    left_probs = _discrete_branch_probabilities(
        distribution,
        branch_index=left_branch_index,
        label="lean_left",
    )
    right_probs = _discrete_branch_probabilities(
        distribution,
        branch_index=right_branch_index,
        label="lean_right",
    )
    if left_probs.shape[1] != 2 or right_probs.shape[1] != 2:
        raise ValueError("Split lean probabilities must be binary")
    return right_probs[:, 1] - left_probs[:, 1]


def _discrete_branch_probabilities(
    distribution: object,
    *,
    branch_index: int,
    label: str,
) -> torch.Tensor:
    branch_distribution = _discrete_branch_distribution(
        distribution,
        branch_index=branch_index,
    )
    raw_probs = getattr(branch_distribution, "probs", None)
    if not isinstance(raw_probs, torch.Tensor):
        raise TypeError(f"Discrete {label} distribution must expose categorical probabilities")
    probs: torch.Tensor = raw_probs
    if probs.ndim != 2:
        raise TypeError(f"Discrete {label} probabilities must be batched")
    return probs


def _discrete_branch_distribution(
    distribution: object,
    *,
    branch_index: int,
) -> object:
    discrete_dist = getattr(distribution, "discrete_dist", None)
    branches = getattr(discrete_dist, "distributions", None)
    if not isinstance(branches, Sequence):
        raise TypeError("Actor regularization requires a hybrid discrete distribution")
    try:
        return branches[branch_index]
    except IndexError as exc:
        raise ValueError("Discrete branch index is outside the distribution") from exc
