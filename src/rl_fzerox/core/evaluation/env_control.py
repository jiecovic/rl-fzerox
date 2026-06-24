# src/rl_fzerox/core/evaluation/env_control.py
"""Access evaluation env controls through Gymnasium wrappers."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from fzerox_emulator.arrays import ActionMask
from rl_fzerox.core.engine_tuning import (
    EngineTuningResetSampler,
    EngineTuningSelectionMode,
)


def set_locked_reset_course(env: object, course_id: str | None) -> None:
    """Lock reset selection even when training observation wrappers are present."""

    _env_control_method(env, "set_locked_reset_course")(course_id)


def set_engine_tuning_sampler(env: object, sampler: EngineTuningResetSampler | None) -> None:
    """Update engine-tuning reset choices through transparent Gym wrappers."""

    _env_control_method(env, "set_engine_tuning_sampler")(sampler)


def set_engine_tuning_selection(env: object, selection: EngineTuningSelectionMode) -> None:
    """Select sampled or greedy engine-tuning resets through transparent wrappers."""

    _env_control_method(env, "set_engine_tuning_selection")(selection)


def action_masks(env: object) -> ActionMask:
    """Return the action mask exposed by the wrapped F-Zero X env."""

    return np.asarray(_env_control_method(env, "action_masks")(), dtype=bool)


def _env_control_method(env: object, method_name: str) -> Callable[..., object]:
    return _env_control_method_from(env, method_name, seen=set())


def _env_control_method_from(
    env: object,
    method_name: str,
    *,
    seen: set[int],
) -> Callable[..., object]:
    env_id = id(env)
    if env_id in seen:
        raise AttributeError(f"evaluation env wrapper cycle while resolving {method_name!r}")
    seen.add(env_id)

    try:
        direct_method = object.__getattribute__(env, method_name)
    except AttributeError:
        direct_method = None
    if callable(direct_method):
        return direct_method

    try:
        inner_env = object.__getattribute__(env, "env")
    except AttributeError:
        inner_env = None
    if inner_env is not None and inner_env is not env:
        return _env_control_method_from(inner_env, method_name, seen=seen)

    raise AttributeError(f"evaluation env does not expose {method_name!r}")
