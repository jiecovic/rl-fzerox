# src/rl_fzerox/core/training/session/model/action_bias.py
from __future__ import annotations

import torch as th
from stable_baselines3.common.vec_env import VecEnv

from rl_fzerox.core.config.schema import PolicyConfig
from rl_fzerox.core.envs.actions import DiscreteActionDimension


def apply_initial_action_biases(
    model: object,
    *,
    train_env: VecEnv,
    policy_config: PolicyConfig,
) -> None:
    """Apply configured one-shot logit nudges to freshly initialized PPO policies."""

    gas_on_logit = float(policy_config.action_bias.gas_on_logit)
    if gas_on_logit == 0.0:
        return

    gas_on_logit_index = _discrete_branch_logit_index(
        _env_action_dimensions(train_env),
        branch_label="gas",
        action_index=1,
    )
    bias = _discrete_head_bias(model)
    if gas_on_logit_index >= bias.numel():
        raise RuntimeError(
            "policy.action_bias.gas_on_logit resolved outside the policy discrete "
            f"logit head: index={gas_on_logit_index}, size={bias.numel()}"
        )

    with th.no_grad():
        bias[gas_on_logit_index] += gas_on_logit


def _env_action_dimensions(train_env: VecEnv) -> tuple[DiscreteActionDimension, ...]:
    try:
        values = train_env.get_attr("action_dimensions")
    except (AttributeError, EOFError, RuntimeError, ValueError) as exc:
        raise RuntimeError(
            "policy.action_bias.gas_on_logit requires env action_dimensions metadata"
        ) from exc
    if not values:
        raise RuntimeError("policy.action_bias.gas_on_logit could not read action dimensions")

    dimensions = values[0]
    if not isinstance(dimensions, tuple) or not all(
        isinstance(dimension, DiscreteActionDimension) for dimension in dimensions
    ):
        raise RuntimeError(
            "policy.action_bias.gas_on_logit requires tuple[DiscreteActionDimension, ...]"
        )
    return dimensions


def _discrete_branch_logit_index(
    dimensions: tuple[DiscreteActionDimension, ...],
    *,
    branch_label: str,
    action_index: int,
) -> int:
    offset = 0
    for dimension in dimensions:
        if dimension.label == branch_label:
            if not 0 <= action_index < dimension.size:
                raise RuntimeError(
                    f"Invalid {branch_label} action bias index {action_index}; "
                    f"valid range is [0, {dimension.size - 1}]"
                )
            return offset + action_index
        offset += dimension.size
    raise RuntimeError(
        f"policy.action_bias.gas_on_logit requires a discrete {branch_label!r} branch"
    )


def _discrete_head_bias(model: object) -> th.Tensor:
    policy = getattr(model, "policy", None)
    action_net = getattr(policy, "action_net", None)
    discrete_net = getattr(action_net, "discrete_net", None)
    bias = getattr(discrete_net, "bias", None)
    if not isinstance(bias, th.Tensor):
        raise RuntimeError(
            "policy.action_bias.gas_on_logit requires a hybrid PPO discrete action head"
        )
    return bias
