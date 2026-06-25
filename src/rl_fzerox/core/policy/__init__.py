# src/rl_fzerox/core/policy/__init__.py
"""Lazy public facade for policy extractors and auxiliary-state helpers.

Training, evaluation, and manager metadata import this package surface. The
facade keeps those imports stable while delaying Torch/SB3X-heavy modules until
a caller requests a concrete extractor or target helper.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rl_fzerox.core.policy.auxiliary_state.targets import (
        auxiliary_state_target_spec,
        auxiliary_state_target_vector,
        auxiliary_state_target_vector_or_zeros,
        auxiliary_state_target_vector_space,
    )
    from rl_fzerox.core.policy.extractors import (
        FZeroXImageStateExtractor,
        FZeroXObservationCnnExtractor,
    )

_EXPORT_MODULES = {
    "FZeroXImageStateExtractor": "rl_fzerox.core.policy.extractors",
    "FZeroXObservationCnnExtractor": "rl_fzerox.core.policy.extractors",
    "auxiliary_state_target_spec": "rl_fzerox.core.policy.auxiliary_state.targets",
    "auxiliary_state_target_vector": "rl_fzerox.core.policy.auxiliary_state.targets",
    "auxiliary_state_target_vector_or_zeros": "rl_fzerox.core.policy.auxiliary_state.targets",
    "auxiliary_state_target_vector_space": "rl_fzerox.core.policy.auxiliary_state.targets",
}

__all__ = [
    "FZeroXImageStateExtractor",
    "FZeroXObservationCnnExtractor",
    "auxiliary_state_target_spec",
    "auxiliary_state_target_vector",
    "auxiliary_state_target_vector_or_zeros",
    "auxiliary_state_target_vector_space",
]


def __getattr__(name: str) -> object:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
