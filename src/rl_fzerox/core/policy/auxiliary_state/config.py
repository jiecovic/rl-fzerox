# src/rl_fzerox/core/policy/auxiliary_state/config.py
"""Parsing helpers for auxiliary-state and actor-regularization config."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from rl_fzerox.core.policy.auxiliary_state.heads import AuxiliaryStateLossTerm
from rl_fzerox.core.policy.auxiliary_state.targets import (
    is_auxiliary_state_target_name,
    resolve_auxiliary_state_target,
)


def _auxiliary_state_loss_terms(
    config: Mapping[str, object] | None,
) -> tuple[AuxiliaryStateLossTerm, ...]:
    if config is None:
        return ()
    losses = config.get("losses")
    if not isinstance(losses, Sequence):
        return ()
    resolved: list[AuxiliaryStateLossTerm] = []
    for entry in losses:
        if not isinstance(entry, Mapping):
            raise TypeError("policy auxiliary loss entries must be mappings")
        name = entry.get("name")
        weight = entry.get("weight", 1.0)
        grounded_only = entry.get("grounded_only", False)
        if not isinstance(name, str):
            raise TypeError("policy auxiliary loss names must be strings")
        if not is_auxiliary_state_target_name(name):
            raise ValueError(f"Unsupported policy auxiliary loss target: {name!r}")
        target = resolve_auxiliary_state_target(name)
        resolved.append(
            AuxiliaryStateLossTerm(
                name=target.name,
                weight=float(weight),
                grounded_only=bool(grounded_only),
            )
        )
    return tuple(resolved)


def _auxiliary_head_arch(config: Mapping[str, object] | None) -> tuple[int, ...]:
    if config is None:
        return ()
    raw_head_arch = config.get("head_arch")
    if not isinstance(raw_head_arch, Sequence):
        return ()
    resolved: list[int] = []
    for value in raw_head_arch:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise TypeError("policy auxiliary head widths must be numeric")
        width = int(value)
        if width <= 0:
            raise ValueError("policy auxiliary head widths must be positive")
        resolved.append(width)
    return tuple(resolved)


def _grounded_pitch_neutral_loss_weight(
    config: Mapping[str, object] | None,
) -> float:
    return _non_negative_config_float(
        config,
        key="grounded_pitch_neutral_loss_weight",
        label="grounded pitch neutral loss weight",
        default=0.0,
    )


def _pitch_std_cap_loss_weight(
    config: Mapping[str, object] | None,
) -> float:
    return _non_negative_config_float(
        config,
        key="pitch_std_cap_loss_weight",
        label="pitch std cap loss weight",
        default=0.0,
    )


def _grounded_pitch_std_cap(
    config: Mapping[str, object] | None,
) -> float:
    return _positive_config_float(
        config,
        key="grounded_pitch_std_cap",
        label="grounded pitch std cap",
        default=0.35,
    )


def _airborne_pitch_std_cap(
    config: Mapping[str, object] | None,
) -> float:
    return _positive_config_float(
        config,
        key="airborne_pitch_std_cap",
        label="airborne pitch std cap",
        default=0.8,
    )


def _steer_std_cap_loss_weight(
    config: Mapping[str, object] | None,
) -> float:
    return _non_negative_config_float(
        config,
        key="steer_std_cap_loss_weight",
        label="steer std cap loss weight",
        default=0.0,
    )


def _steer_std_cap(
    config: Mapping[str, object] | None,
) -> float:
    return _positive_config_float(
        config,
        key="steer_std_cap",
        label="steer std cap",
        default=1.0,
    )


def _steer_signed_balance_loss_weight(
    config: Mapping[str, object] | None,
) -> float:
    return _non_negative_config_float(
        config,
        key="steer_signed_balance_loss_weight",
        label="steer signed balance loss weight",
        default=0.0,
    )


def _steer_signed_balance_deadzone(
    config: Mapping[str, object] | None,
) -> float:
    return _non_negative_config_float(
        config,
        key="steer_signed_balance_deadzone",
        label="steer signed balance deadzone",
        default=0.2,
    )


def _lean_signed_balance_loss_weight(
    config: Mapping[str, object] | None,
) -> float:
    return _non_negative_config_float(
        config,
        key="lean_signed_balance_loss_weight",
        label="lean signed balance loss weight",
        default=0.0,
    )


def _lean_signed_balance_deadzone(
    config: Mapping[str, object] | None,
) -> float:
    return _non_negative_config_float(
        config,
        key="lean_signed_balance_deadzone",
        label="lean signed balance deadzone",
        default=0.1,
    )


def _non_negative_config_float(
    config: Mapping[str, object] | None,
    *,
    key: str,
    label: str,
    default: float,
) -> float:
    if config is None:
        return default
    value = config.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{label} must be numeric")
    resolved = float(value)
    if resolved < 0.0:
        raise ValueError(f"{label} must be non-negative")
    return resolved


def _positive_config_float(
    config: Mapping[str, object] | None,
    *,
    key: str,
    label: str,
    default: float,
) -> float:
    if config is None:
        return default
    value = config.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{label} must be numeric")
    resolved = float(value)
    if resolved <= 0.0:
        raise ValueError(f"{label} must be positive")
    return resolved


def _axis_index(names: Sequence[str], axis: str) -> int | None:
    try:
        return tuple(names).index(axis)
    except ValueError:
        return None


def _pitch_bucket_values(bucket_count: int) -> tuple[float, ...]:
    neutral_index = int(bucket_count) // 2
    if neutral_index <= 0:
        return (0.0,)
    return tuple(
        float(index - neutral_index) / float(neutral_index) for index in range(int(bucket_count))
    )
