# src/rl_fzerox/core/training/session/model/action_bias.py
"""Policy-head bias initialization derived from configured action priors."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from zipfile import BadZipFile, ZipFile

import torch as th

from rl_fzerox.core.envs.actions import DiscreteActionDimension
from rl_fzerox.core.runtime_spec.schema import PolicyConfig

MODEL_ACTION_BIAS_OFFSETS_ATTR = "_fzerox_action_bias_logit_offsets"


@dataclass(frozen=True, slots=True)
class _ActionBiasSpec:
    field_name: str
    branch_label: str
    action_index: int
    value: float


@dataclass(frozen=True, slots=True)
class _ActionBiasTarget:
    field_name: str
    branch_label: str
    action_index: int


ACTION_BIAS_TARGETS = (
    _ActionBiasTarget(field_name="gas_on_logit", branch_label="gas", action_index=1),
    _ActionBiasTarget(field_name="air_brake_on_logit", branch_label="air_brake", action_index=1),
    _ActionBiasTarget(field_name="spin_idle_logit", branch_label="spin", action_index=0),
)
ACTION_BIAS_FIELD_NAMES = tuple(target.field_name for target in ACTION_BIAS_TARGETS)


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


def apply_weights_only_action_bias_delta(
    model: object,
    *,
    train_env: TrainingEnvActionDimensions,
    policy_config: PolicyConfig,
    source_offsets: dict[str, float],
) -> None:
    """Apply fork config logit nudges on top of source checkpoint metadata.

    Checkpoint metadata stores the cumulative bias already baked into the
    source weights. Fork config fields are launch-time deltas, so a reset fork
    config of zero must preserve the source bias instead of undoing it.
    """

    base_offsets = _normalized_action_bias_offsets(
        source_offsets,
        source="weights-only resume source checkpoint",
    )
    requested_deltas = _configured_action_bias_offsets(policy_config)
    desired_offsets = {
        field_name: base_offsets[field_name] + requested_deltas[field_name]
        for field_name in ACTION_BIAS_FIELD_NAMES
    }
    previous_offsets = _model_action_bias_offsets(model)
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


def load_action_bias_offsets_from_archive(model_path: Path) -> dict[str, float]:
    """Read persisted action-bias metadata from an SB3 model archive."""

    try:
        with ZipFile(model_path) as archive:
            data = archive.read("data")
    except (BadZipFile, FileNotFoundError, KeyError) as exc:
        raise _action_bias_metadata_error("weights-only resume source checkpoint") from exc

    try:
        loaded: object = json.loads(data.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise _action_bias_metadata_error("weights-only resume source checkpoint") from exc
    if not isinstance(loaded, dict):
        raise _action_bias_metadata_error("weights-only resume source checkpoint")
    return _normalized_action_bias_offsets(
        loaded.get(MODEL_ACTION_BIAS_OFFSETS_ATTR),
        source="weights-only resume source checkpoint",
    )


def set_model_action_bias_offsets(model: object, offsets: dict[str, float]) -> None:
    _set_model_action_bias_offsets(model, offsets)


def _configured_action_bias_offsets(policy_config: PolicyConfig) -> dict[str, float]:
    return {
        target.field_name: float(getattr(policy_config.action_bias, target.field_name))
        for target in ACTION_BIAS_TARGETS
    }


def _model_action_bias_offsets(model: object) -> dict[str, float]:
    return _normalized_action_bias_offsets(
        getattr(model, MODEL_ACTION_BIAS_OFFSETS_ATTR, None),
        source="full-model resume checkpoint",
    )


def _normalized_action_bias_offsets(value: object, *, source: str) -> dict[str, float]:
    if value is None or not isinstance(value, dict):
        raise _action_bias_metadata_error(source)
    offsets: dict[str, float] = {}
    for field_name in ACTION_BIAS_FIELD_NAMES:
        raw_offset = value.get(field_name)
        if isinstance(raw_offset, int | float) and not isinstance(raw_offset, bool):
            offsets[field_name] = float(raw_offset)
            continue
        raise _action_bias_metadata_error(source)
    return offsets


def _action_bias_metadata_error(source: str) -> RuntimeError:
    return RuntimeError(
        f"{source} is missing action-bias metadata; "
        "run the local action-bias checkpoint migration before resuming"
    )


def _set_model_action_bias_offsets(model: object, offsets: dict[str, float]) -> None:
    setattr(model, MODEL_ACTION_BIAS_OFFSETS_ATTR, dict(offsets))


def _action_bias_specs(offsets: dict[str, float]) -> tuple[_ActionBiasSpec, ...]:
    return (
        *(
            _ActionBiasSpec(
                field_name=target.field_name,
                branch_label=target.branch_label,
                action_index=target.action_index,
                value=offsets[target.field_name],
            )
            for target in ACTION_BIAS_TARGETS
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
