# src/rl_fzerox/core/training/inference/loader.py
from __future__ import annotations

import inspect
from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from typing import Protocol, cast

import numpy as np

from rl_fzerox.core.envs.observations import ObservationValue


class _HasPredict(Protocol):
    predict: Callable[..., tuple[object, object]]


def _load_saved_policy(policy_path: Path, *, run_dir: Path | None = None):
    """Load one saved policy-only SB3 artifact."""

    import torch
    from gymnasium import spaces

    _ensure_policy_dependencies_loaded()

    saved_policy = torch.load(policy_path, map_location="cpu", weights_only=False)
    saved_data = saved_policy.get("data", {})
    observation_space = saved_data.get("observation_space")
    policy_classes = _policy_classes_for_algorithm(
        algorithm=_load_saved_policy_algorithm(run_dir),
    )
    CnnPolicy, MultiInputPolicy = policy_classes
    policy_class = MultiInputPolicy if isinstance(observation_space, spaces.Dict) else CnnPolicy
    return policy_class.load(str(policy_path), device="auto")


def _policy_mtime_ns(policy_path: Path) -> int:
    return policy_path.stat().st_mtime_ns


def _ensure_policy_dependencies_loaded() -> None:
    """Import custom policy modules before SB3 deserializes saved artifacts."""

    import_module("rl_fzerox.core.policy.extractors")


def _policy_predict_fn(policy: object) -> Callable[..., tuple[object, object]]:
    return cast(_HasPredict, policy).predict


def _predict_policy_action(
    policy: object,
    observation: ObservationValue,
    *,
    deterministic: bool,
    action_masks: np.ndarray | None,
) -> tuple[object, object]:
    predict = _policy_predict_fn(policy)
    if action_masks is None:
        return predict(observation, deterministic=deterministic)
    return predict(observation, deterministic=deterministic, action_masks=action_masks)


def _policy_supports_action_masks(policy: object) -> bool:
    return "action_masks" in inspect.signature(_policy_predict_fn(policy)).parameters


def _policy_classes_for_algorithm(*, algorithm: str):
    if algorithm == "maskable_ppo":
        from sb3_contrib.ppo_mask import CnnPolicy, MultiInputPolicy

        return CnnPolicy, MultiInputPolicy

    from stable_baselines3.ppo import CnnPolicy, MultiInputPolicy

    return CnnPolicy, MultiInputPolicy


def _load_saved_policy_algorithm(run_dir: Path | None) -> str:
    if run_dir is None:
        return "ppo"

    config_path = run_dir / "train_config.yaml"
    if not config_path.is_file():
        return "ppo"

    try:
        from rl_fzerox.core.config import load_train_app_config
        from rl_fzerox.core.training.session.model import (
            resolve_effective_training_algorithm,
            training_requires_action_masks,
        )

        config = load_train_app_config(config_path)
        return resolve_effective_training_algorithm(
            train_config=config.train,
            masking_required=training_requires_action_masks(config),
        )
    except Exception:
        return "ppo"
