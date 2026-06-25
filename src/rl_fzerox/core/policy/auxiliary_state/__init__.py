# src/rl_fzerox/core/policy/auxiliary_state/__init__.py
"""Lazy facade for auxiliary-state target metadata and observation helpers."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rl_fzerox.core.policy.auxiliary_state.names import AuxiliaryStateTargetName
    from rl_fzerox.core.policy.auxiliary_state.observations import (
        auxiliary_state_targets_field,
        auxiliary_state_targets_from_mapping,
        mapping_has_auxiliary_state_targets,
        mapping_with_auxiliary_state_targets,
    )
    from rl_fzerox.core.policy.auxiliary_state.targets import (
        auxiliary_state_target_bounds,
        auxiliary_state_target_spec,
        auxiliary_state_target_values,
        auxiliary_state_target_vector,
        auxiliary_state_target_vector_or_zeros,
        auxiliary_state_target_vector_space,
        resolve_auxiliary_state_target,
        supported_auxiliary_state_target_names,
    )

_EXPORT_MODULES = {
    "AuxiliaryStateTargetName": "rl_fzerox.core.policy.auxiliary_state.names",
    "auxiliary_state_targets_field": "rl_fzerox.core.policy.auxiliary_state.observations",
    "auxiliary_state_targets_from_mapping": "rl_fzerox.core.policy.auxiliary_state.observations",
    "mapping_has_auxiliary_state_targets": "rl_fzerox.core.policy.auxiliary_state.observations",
    "mapping_with_auxiliary_state_targets": "rl_fzerox.core.policy.auxiliary_state.observations",
    "auxiliary_state_target_bounds": "rl_fzerox.core.policy.auxiliary_state.targets",
    "auxiliary_state_target_spec": "rl_fzerox.core.policy.auxiliary_state.targets",
    "auxiliary_state_target_values": "rl_fzerox.core.policy.auxiliary_state.targets",
    "auxiliary_state_target_vector": "rl_fzerox.core.policy.auxiliary_state.targets",
    "auxiliary_state_target_vector_or_zeros": "rl_fzerox.core.policy.auxiliary_state.targets",
    "auxiliary_state_target_vector_space": "rl_fzerox.core.policy.auxiliary_state.targets",
    "resolve_auxiliary_state_target": "rl_fzerox.core.policy.auxiliary_state.targets",
    "supported_auxiliary_state_target_names": "rl_fzerox.core.policy.auxiliary_state.targets",
}

__all__ = [
    "AuxiliaryStateTargetName",
    "auxiliary_state_target_bounds",
    "auxiliary_state_targets_field",
    "auxiliary_state_targets_from_mapping",
    "auxiliary_state_target_spec",
    "auxiliary_state_target_values",
    "auxiliary_state_target_vector",
    "auxiliary_state_target_vector_or_zeros",
    "auxiliary_state_target_vector_space",
    "mapping_has_auxiliary_state_targets",
    "mapping_with_auxiliary_state_targets",
    "resolve_auxiliary_state_target",
    "supported_auxiliary_state_target_names",
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
