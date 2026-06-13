# src/rl_fzerox/core/policy/auxiliary_state/target_tensors.py
from __future__ import annotations

from collections.abc import Mapping

import torch
from stable_baselines3.common.type_aliases import PyTorchObs

from rl_fzerox.core.policy.auxiliary_state.observations import (
    auxiliary_state_targets_field,
)


def _require_auxiliary_targets(obs: PyTorchObs) -> torch.Tensor:
    if not isinstance(obs, Mapping):
        raise TypeError("Auxiliary-state policies require dict observations")
    field_name = auxiliary_state_targets_field()
    aux_targets = obs.get(field_name)
    if not isinstance(aux_targets, torch.Tensor):
        raise TypeError(f"Auxiliary-state policies require tensor observation key {field_name!r}")
    aux_target_tensor: torch.Tensor = aux_targets
    return aux_target_tensor.float().flatten(start_dim=1)


def _optional_auxiliary_targets(obs: PyTorchObs) -> torch.Tensor | None:
    if not isinstance(obs, Mapping):
        return None
    field_name = auxiliary_state_targets_field()
    aux_targets = obs.get(field_name)
    if not isinstance(aux_targets, torch.Tensor):
        return None
    aux_target_tensor: torch.Tensor = aux_targets
    return aux_target_tensor.float().flatten(start_dim=1)
