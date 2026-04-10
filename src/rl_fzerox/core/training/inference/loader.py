# src/rl_fzerox/core/training/inference/loader.py
from __future__ import annotations

import inspect
from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from typing import Protocol, TypeGuard

import numpy as np

from rl_fzerox.core.envs.observations import ObservationValue

PolicyState = tuple[np.ndarray, ...] | None


class _HasPredict(Protocol):
    def predict(
        self,
        observation: ObservationValue,
        state: PolicyState = None,
        episode_start: np.ndarray | None = None,
        deterministic: bool = True,
    ) -> tuple[object, PolicyState]: ...


class _HasMaskablePredict(Protocol):
    def predict(
        self,
        observation: ObservationValue,
        state: PolicyState = None,
        episode_start: np.ndarray | None = None,
        deterministic: bool = True,
        action_masks: np.ndarray | None = None,
    ) -> tuple[object, PolicyState]: ...


def _load_saved_policy(
    policy_path: Path,
    *,
    run_dir: Path | None = None,
    device: str = "cpu",
) -> _HasPredict:
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
    loaded_policy = policy_class.load(str(policy_path), device=device)
    if not _has_predict(loaded_policy):
        raise TypeError("Loaded policy does not expose a compatible predict(...) method")
    return loaded_policy


def _policy_mtime_ns(policy_path: Path) -> int:
    return policy_path.stat().st_mtime_ns


def _ensure_policy_dependencies_loaded() -> None:
    """Import custom policy modules before SB3 deserializes saved artifacts."""

    import_module("rl_fzerox.core.policy.extractors")


def _policy_predict_fn(policy: object) -> Callable[..., tuple[object, PolicyState]]:
    if not _has_predict(policy):
        raise TypeError("Policy does not expose a compatible predict(...) method")
    return policy.predict


def _predict_policy_action(
    policy: object,
    observation: ObservationValue,
    *,
    state: PolicyState,
    episode_start: np.ndarray | None,
    deterministic: bool,
    action_masks: np.ndarray | None,
) -> tuple[object, PolicyState]:
    if action_masks is None:
        return _policy_predict_fn(policy)(
            observation,
            state=state,
            episode_start=episode_start,
            deterministic=deterministic,
        )
    if not _has_maskable_predict(policy):
        raise TypeError("Policy does not support masked action prediction")
    return policy.predict(
        observation,
        state=state,
        episode_start=episode_start,
        deterministic=deterministic,
        action_masks=action_masks,
    )


def _policy_supports_action_masks(policy: object) -> bool:
    return "action_masks" in inspect.signature(_policy_predict_fn(policy)).parameters


def _has_predict(policy: object) -> TypeGuard[_HasPredict]:
    predict = getattr(policy, "predict", None)
    return callable(predict)


def _has_maskable_predict(policy: object) -> TypeGuard[_HasMaskablePredict]:
    if not _has_predict(policy):
        return False
    return "action_masks" in inspect.signature(policy.predict).parameters


def _policy_classes_for_algorithm(*, algorithm: str):
    if algorithm == "maskable_recurrent_ppo":
        try:
            from sb3x.ppo_mask_recurrent import CnnLstmPolicy, MultiInputLstmPolicy
        except ImportError as exc:
            raise RuntimeError(
                "Loading recurrent checkpoints requires sb3x in the active environment."
            ) from exc

        return CnnLstmPolicy, MultiInputLstmPolicy
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

        config = load_train_app_config(config_path)
        if config.train.algorithm in {"maskable_ppo", "maskable_recurrent_ppo"}:
            return config.train.algorithm
        # Historical runs may have saved `auto` from the pre-maskable default
        # era, so only explicit algorithm values are treated as non-legacy here.
        return "ppo"
    except Exception:
        return "ppo"
