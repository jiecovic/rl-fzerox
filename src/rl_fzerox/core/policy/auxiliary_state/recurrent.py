# src/rl_fzerox/core/policy/auxiliary_state/recurrent.py
from __future__ import annotations

import numpy as np
import torch
from sb3x.common.recurrent import count_vectorized_envs
from sb3x.ppo_mask_hybrid_recurrent.policies import (
    MaskableHybridRecurrentMultiInputActorCriticPolicy,
)
from stable_baselines3.common.type_aliases import PyTorchObs

from fzerox_emulator.arrays import BoolArray, PolicyState


def _recurrent_tensor_state(
    *,
    policy: MaskableHybridRecurrentMultiInputActorCriticPolicy,
    state: PolicyState,
    obs: PyTorchObs,
    episode_start: BoolArray | None,
) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    n_envs = count_vectorized_envs(obs)
    if state is None:
        zeros = np.concatenate(
            [np.zeros(policy.lstm_hidden_state_shape) for _ in range(n_envs)],
            axis=1,
        )
        state = (zeros, zeros)
    if episode_start is None:
        episode_start = np.zeros(n_envs, dtype=bool)

    return (
        (
            torch.Tensor(state[0]).to(device=policy.device),
            torch.Tensor(state[1]).to(device=policy.device),
        ),
        torch.Tensor(episode_start).to(device=policy.device),
    )
