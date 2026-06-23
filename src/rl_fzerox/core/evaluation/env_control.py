# src/rl_fzerox/core/evaluation/env_control.py
"""Access evaluation env controls through Gymnasium wrappers."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from fzerox_emulator.arrays import ActionMask


def set_locked_reset_course(env: object, course_id: str | None) -> None:
    """Lock reset selection even when training observation wrappers are present."""

    _env_control_method(env, "set_locked_reset_course")(course_id)


def sync_checkpoint_curriculum_stage(env: object, stage_index: int | None) -> None:
    """Apply checkpoint-stage action masks through transparent Gym wrappers."""

    _env_control_method(env, "sync_checkpoint_curriculum_stage")(stage_index)


def action_masks(env: object) -> ActionMask:
    """Return the action mask exposed by the wrapped F-Zero X env."""

    return np.asarray(_env_control_method(env, "action_masks")(), dtype=bool)


def _env_control_method(env: object, method_name: str) -> Callable[..., object]:
    direct_method = getattr(env, method_name, None)
    if callable(direct_method):
        return direct_method

    get_wrapper_attr = getattr(env, "get_wrapper_attr", None)
    if callable(get_wrapper_attr):
        wrapper_method = get_wrapper_attr(method_name)
        if callable(wrapper_method):
            return wrapper_method

    raise AttributeError(f"evaluation env does not expose {method_name!r}")
