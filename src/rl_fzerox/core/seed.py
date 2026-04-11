# src/rl_fzerox/core/seed.py
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

_MASK64 = (1 << 64) - 1
_SPLITMIX64_GAMMA = 0x9E3779B97F4A7C15


@dataclass(frozen=True, slots=True)
class _SeedDomains:
    """Fixed domain separators used to derive independent process RNG seeds."""

    python: int
    numpy: int


_SEED_DOMAINS = _SeedDomains(
    python=0xB5AD4ECEDA1CE2A9,
    numpy=0x94D049BB133111EB,
)


def normalize_seed(seed: int) -> int:
    """Normalize a Python integer into an unsigned 64-bit seed."""

    return seed & _MASK64


def splitmix64(value: int) -> int:
    """Mix a 64-bit integer using the SplitMix64 output function."""

    z = (normalize_seed(value) + _SPLITMIX64_GAMMA) & _MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
    return (z ^ (z >> 31)) & _MASK64


def derive_seed(master_seed: int | None, *components: int) -> int | None:
    """Derive a stable 64-bit seed with explicit domain separation."""

    if master_seed is None:
        return None

    derived = normalize_seed(master_seed)
    for component in components:
        derived = splitmix64(derived ^ normalize_seed(component))
    return derived


def seed_process(seed: int | None) -> None:
    """Seed Python-side randomness for the current process."""

    if seed is None:
        return

    python_seed = derive_seed(seed, _SEED_DOMAINS.python)
    numpy_seed = derive_seed(seed, _SEED_DOMAINS.numpy)
    if python_seed is None or numpy_seed is None:
        return

    random.seed(python_seed)
    np.random.seed(np.uint32(numpy_seed & 0xFFFFFFFF))
