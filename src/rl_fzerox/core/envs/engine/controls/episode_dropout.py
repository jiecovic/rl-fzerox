# src/rl_fzerox/core/envs/engine/controls/episode_dropout.py
from __future__ import annotations

import random
from collections.abc import Container
from dataclasses import dataclass
from typing import Literal

from rl_fzerox.core.seed import derive_seed

EpisodeActionMaskBranch = Literal["lean", "air_brake", "spin"]


@dataclass(frozen=True, slots=True)
class _EpisodeActionMaskSeedSubstreams:
    """Substream ids inside the already domain-separated episode-mask RNG seed."""

    air_brake: int = 1
    spin: int = 2


_EPISODE_ACTION_MASK_SEED_SUBSTREAMS = _EpisodeActionMaskSeedSubstreams()


@dataclass(frozen=True, slots=True)
class EpisodeActionMasks:
    """Episode-scoped action branches sampled into neutral-only masks."""

    lean: bool = False
    air_brake: bool = False
    spin: bool = False


def sample_episode_action_masks(
    *,
    lean_probability: float,
    air_brake_probability: float,
    spin_probability: float,
    seed: int | None,
    available_branches: Container[str] | None = None,
) -> EpisodeActionMasks:
    """Sample all episode-scoped action-dropout masks from one reset seed."""

    return EpisodeActionMasks(
        lean=sample_episode_action_mask(
            probability=lean_probability,
            seed=seed,
            branch="lean",
        ),
        air_brake=sample_episode_action_mask(
            probability=_branch_probability(
                air_brake_probability,
                branch="air_brake",
                available_branches=available_branches,
            ),
            seed=seed,
            branch="air_brake",
        ),
        spin=sample_episode_action_mask(
            probability=_branch_probability(
                spin_probability,
                branch="spin",
                available_branches=available_branches,
            ),
            seed=seed,
            branch="spin",
        ),
    )


def sample_episode_lean_mask(*, probability: float, seed: int | None) -> bool:
    """Sample whether lean should be masked for one whole episode."""

    return sample_episode_action_mask(probability=probability, seed=seed, branch="lean")


def sample_episode_action_mask(
    *,
    probability: float,
    seed: int | None,
    branch: EpisodeActionMaskBranch,
) -> bool:
    """Sample whether one action branch should be masked for one whole episode."""

    if probability <= 0.0:
        return False
    if probability >= 1.0:
        return True
    if seed is None:
        return random.random() < probability
    return random.Random(_branch_seed(seed, branch)).random() < probability


def _branch_seed(seed: int, branch: EpisodeActionMaskBranch) -> int:
    if branch == "lean":
        return seed
    if branch == "air_brake":
        derived = derive_seed(seed, _EPISODE_ACTION_MASK_SEED_SUBSTREAMS.air_brake)
    else:
        derived = derive_seed(seed, _EPISODE_ACTION_MASK_SEED_SUBSTREAMS.spin)
    if derived is None:
        raise RuntimeError("episode action mask seed derivation unexpectedly returned None")
    return derived


def _branch_probability(
    probability: float,
    *,
    branch: EpisodeActionMaskBranch,
    available_branches: Container[str] | None,
) -> float:
    if available_branches is not None and branch not in available_branches:
        return 0.0
    return probability
