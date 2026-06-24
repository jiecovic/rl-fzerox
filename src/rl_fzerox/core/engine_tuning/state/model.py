# src/rl_fzerox/core/engine_tuning/state/model.py
"""Persisted learned-model state for experimental engine tuners."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from rl_fzerox.core.engine_tuning.types import EngineTunerBackend


@dataclass(frozen=True, slots=True)
class EngineTuningTensorState:
    """One tensor stored in the engine-tuner model checkpoint."""

    name: str
    value: torch.Tensor


@dataclass(frozen=True, slots=True)
class EngineTuningEnsembleMemberState:
    """One persisted MLP ensemble member."""

    tensors: tuple[EngineTuningTensorState, ...]


@dataclass(frozen=True, slots=True)
class EngineTuningModelContextState:
    """Observed context metadata for model-backed tuners."""

    context_key: str
    course_key: str
    vehicle_id: str
    finish_count: int = 0


@dataclass(frozen=True, slots=True)
class EngineTuningModelState:
    """Optional learned model state for non-aggregate tuner backends."""

    backend: EngineTunerBackend
    course_keys: tuple[str, ...]
    vehicle_ids: tuple[str, ...]
    members: tuple[EngineTuningEnsembleMemberState, ...] = ()
    contexts: tuple[EngineTuningModelContextState, ...] = ()
