# src/rl_fzerox/core/training/session/model/action_bias.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch as th

from rl_fzerox.core.envs.actions import DiscreteActionDimension
from rl_fzerox.core.runtime_spec.schema import PolicyConfig

MODEL_ACTION_BIAS_OFFSETS_ATTR = "_fzerox_action_bias_logit_offsets"
_ACTION_BIAS_FIELD_NAMES = ("gas_on_logit", "spin_idle_logit")
_MARKERLESS_FULL_RESUME_DELTA_FIELDS = frozenset(("spin_idle_logit",))


@dataclass(frozen=True, slots=True)
class _ActionBiasSpec:
    field_name: str
    branch_label: str
    action_index: int
    value: float


class TrainingEnvActionDimensions(Protocol):
    def get_attr(self, attr_name: str) -> list[object]: ...


def apply_initial_action_biases(
    model: object,
    *,
    train_env: TrainingEnvActionDimensions,
    policy_config: PolicyConfig,
) -> None:
    """Apply configured one-shot logit nudges to freshly initialized PPO policies."""

    offsets = _configured_action_bias_offsets(policy_config)
    _apply_action_bias_specs(
        model,
        train_env=train_env,
        specs=_action_bias_specs(offsets),
    )
    _set_model_action_bias_offsets(model, offsets)


def apply_resume_action_bias_delta(
    model: object,
    *,
    train_env: TrainingEnvActionDimensions,
    policy_config: PolicyConfig,
) -> None:
    """Apply only newly requested logit nudges after a full-model resume."""

    desired_offsets = _configured_action_bias_offsets(policy_config)
    previous_offsets = _model_action_bias_offsets(model, desired_offsets=desired_offsets)
    delta_offsets = {
        field_name: desired_offsets[field_name] - previous_offsets.get(field_name, 0.0)
        for field_name in desired_offsets
    }
    _apply_action_bias_specs(
        model,
        train_env=train_env,
        specs=_action_bias_specs(delta_offsets),
    )
    _set_model_action_bias_offsets(model, desired_offsets)


def _configured_action_bias_offsets(policy_config: PolicyConfig) -> dict[str, float]:
    return {
        "gas_on_logit": float(policy_config.action_bias.gas_on_logit),
        "spin_idle_logit": float(policy_config.action_bias.spin_idle_logit),
    }


def _model_action_bias_offsets(
    model: object,
    *,
    desired_offsets: dict[str, float],
) -> dict[str, float]:
    value = getattr(model, MODEL_ACTION_BIAS_OFFSETS_ATTR, None)
    if value is None or not isinstance(value, dict):
        # Older checkpoints have no marker. Gas bias already existed as an initial
        # model-construction nudge, so do not replay it. The spin-idle field is new
        # and intentionally resume-adjustable for existing spin-enabled runs.
        return {
            field_name: (
                0.0
                if field_name in _MARKERLESS_FULL_RESUME_DELTA_FIELDS
                else desired_offsets[field_name]
            )
            for field_name in _ACTION_BIAS_FIELD_NAMES
        }
    offsets: dict[str, float] = {}
    for field_name in _ACTION_BIAS_FIELD_NAMES:
        raw_offset = value.get(field_name)
        if isinstance(raw_offset, int | float) and not isinstance(raw_offset, bool):
            offsets[field_name] = float(raw_offset)
    return offsets


def _set_model_action_bias_offsets(model: object, offsets: dict[str, float]) -> None:
    setattr(model, MODEL_ACTION_BIAS_OFFSETS_ATTR, dict(offsets))


def _action_bias_specs(offsets: dict[str, float]) -> tuple[_ActionBiasSpec, ...]:
    return (
        _ActionBiasSpec(
            field_name="gas_on_logit",
            branch_label="gas",
            action_index=1,
            value=offsets["gas_on_logit"],
        ),
        _ActionBiasSpec(
            field_name="spin_idle_logit",
            branch_label="spin",
            action_index=0,
            value=offsets["spin_idle_logit"],
        ),
    )


def _apply_action_bias_specs(
    model: object,
    *,
    train_env: TrainingEnvActionDimensions,
    specs: tuple[_ActionBiasSpec, ...],
) -> None:
    active_specs = tuple(spec for spec in specs if spec.value != 0.0)
    if not active_specs:
        return

    dimensions = _env_action_dimensions(train_env, field_name=active_specs[0].field_name)
    bias = _discrete_head_bias(model, field_name=active_specs[0].field_name)
    for spec in active_specs:
        logit_index = _discrete_branch_logit_index(
            dimensions,
            branch_label=spec.branch_label,
            action_index=spec.action_index,
            field_name=spec.field_name,
        )
        if logit_index >= bias.numel():
            raise RuntimeError(
                f"policy.action_bias.{spec.field_name} resolved outside the policy discrete "
                f"logit head: index={logit_index}, size={bias.numel()}"
            )

    with th.no_grad():
        for spec in active_specs:
            logit_index = _discrete_branch_logit_index(
                dimensions,
                branch_label=spec.branch_label,
                action_index=spec.action_index,
                field_name=spec.field_name,
            )
            bias[logit_index] += spec.value


def _env_action_dimensions(
    train_env: TrainingEnvActionDimensions,
    *,
    field_name: str,
) -> tuple[DiscreteActionDimension, ...]:
    try:
        values = train_env.get_attr("action_dimensions")
    except (AttributeError, EOFError, RuntimeError, ValueError) as exc:
        raise RuntimeError(
            f"policy.action_bias.{field_name} requires env action_dimensions metadata"
        ) from exc
    if not values:
        raise RuntimeError(f"policy.action_bias.{field_name} could not read action dimensions")

    dimensions = values[0]
    if not isinstance(dimensions, tuple) or not all(
        isinstance(dimension, DiscreteActionDimension) for dimension in dimensions
    ):
        raise RuntimeError(
            f"policy.action_bias.{field_name} requires tuple[DiscreteActionDimension, ...]"
        )
    return dimensions


def _discrete_branch_logit_index(
    dimensions: tuple[DiscreteActionDimension, ...],
    *,
    branch_label: str,
    action_index: int,
    field_name: str,
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
        f"policy.action_bias.{field_name} requires a discrete {branch_label!r} branch"
    )


def _discrete_head_bias(model: object, *, field_name: str) -> th.Tensor:
    policy = getattr(model, "policy", None)
    action_net = getattr(policy, "action_net", None)
    discrete_net = getattr(action_net, "discrete_net", None)
    bias = getattr(discrete_net, "bias", None)
    if not isinstance(bias, th.Tensor):
        raise RuntimeError(
            f"policy.action_bias.{field_name} requires a hybrid PPO discrete action head"
        )
    return bias
