# src/rl_fzerox/core/training/inference/loader.py
from __future__ import annotations

import inspect
from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from typing import Protocol, TypeGuard

import numpy as np

from rl_fzerox.core.domain.training_algorithms import (
    FULL_MODEL_POLICY_ALGORITHMS,
    LEGACY_PPO_ALGORITHMS,
    SAVED_POLICY_ALGORITHMS,
    TRAIN_ALGORITHM_HYBRID_ACTION_PPO,
    TRAIN_ALGORITHM_HYBRID_RECURRENT_PPO,
    TRAIN_ALGORITHM_MASKABLE_HYBRID_ACTION_PPO,
    TRAIN_ALGORITHM_MASKABLE_HYBRID_RECURRENT_PPO,
    TRAIN_ALGORITHM_MASKABLE_PPO,
    TRAIN_ALGORITHM_MASKABLE_RECURRENT_PPO,
    TRAIN_ALGORITHM_PPO,
    TRAIN_ALGORITHM_SAC,
)
from rl_fzerox.core.envs.actions import ActionValue
from rl_fzerox.core.envs.observations import ObservationValue
from rl_fzerox.core.training.runs import RUN_LAYOUT, resolve_model_artifact_path

PolicyState = tuple[np.ndarray, ...] | None


class _HasPredict(Protocol):
    def predict(
        self,
        observation: ObservationValue,
        state: PolicyState = None,
        episode_start: np.ndarray | None = None,
        deterministic: bool = True,
    ) -> tuple[ActionValue, PolicyState]: ...


class _HasMaskablePredict(Protocol):
    def predict(
        self,
        observation: ObservationValue,
        state: PolicyState = None,
        episode_start: np.ndarray | None = None,
        deterministic: bool = True,
        action_masks: np.ndarray | None = None,
    ) -> tuple[ActionValue, PolicyState]: ...


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

    algorithm = _load_saved_policy_algorithm(run_dir)
    if algorithm in FULL_MODEL_POLICY_ALGORITHMS:
        if run_dir is None:
            raise RuntimeError(f"{algorithm} policy loading requires the source run directory")
        algorithm_class = _full_model_class_for_algorithm(algorithm)
        model_path = resolve_model_artifact_path(
            run_dir,
            artifact=_artifact_kind_from_policy_path(policy_path),
        )
        loaded_model = algorithm_class.load(str(model_path), device=device)
        if not _has_predict(loaded_model):
            raise TypeError("Loaded model does not expose a compatible predict(...)")
        return loaded_model

    saved_policy = torch.load(policy_path, map_location="cpu", weights_only=False)
    saved_data = saved_policy.get("data", {})
    observation_space = saved_data.get("observation_space")
    policy_classes = _policy_classes_for_algorithm(algorithm=algorithm)
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


def _policy_predict_fn(policy: object) -> Callable[..., tuple[ActionValue, PolicyState]]:
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
) -> tuple[ActionValue, PolicyState]:
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
    if algorithm == TRAIN_ALGORITHM_MASKABLE_HYBRID_ACTION_PPO:
        try:
            from sb3x.ppo_mask_hybrid_action import CnnPolicy, MultiInputPolicy
        except ImportError as exc:
            raise RuntimeError(
                "Loading maskable hybrid action PPO checkpoints requires sb3x "
                "in the active environment."
            ) from exc

        return CnnPolicy, MultiInputPolicy
    if algorithm == TRAIN_ALGORITHM_HYBRID_ACTION_PPO:
        try:
            from sb3x.ppo_hybrid_action import CnnPolicy, MultiInputPolicy
        except ImportError as exc:
            raise RuntimeError(
                "Loading hybrid action PPO checkpoints requires sb3x in the active environment."
            ) from exc

        return CnnPolicy, MultiInputPolicy
    if algorithm == TRAIN_ALGORITHM_HYBRID_RECURRENT_PPO:
        try:
            from sb3x.ppo_hybrid_recurrent import CnnLstmPolicy, MultiInputLstmPolicy
        except ImportError as exc:
            raise RuntimeError(
                "Loading hybrid recurrent PPO checkpoints requires sb3x in the active environment."
            ) from exc

        return CnnLstmPolicy, MultiInputLstmPolicy
    if algorithm == TRAIN_ALGORITHM_MASKABLE_HYBRID_RECURRENT_PPO:
        try:
            from sb3x.ppo_mask_hybrid_recurrent import CnnLstmPolicy, MultiInputLstmPolicy
        except ImportError as exc:
            raise RuntimeError(
                "Loading maskable hybrid recurrent PPO checkpoints requires sb3x "
                "in the active environment."
            ) from exc

        return CnnLstmPolicy, MultiInputLstmPolicy
    if algorithm == TRAIN_ALGORITHM_MASKABLE_RECURRENT_PPO:
        try:
            from sb3x.ppo_mask_recurrent import CnnLstmPolicy, MultiInputLstmPolicy
        except ImportError as exc:
            raise RuntimeError(
                "Loading recurrent checkpoints requires sb3x in the active environment."
            ) from exc

        return CnnLstmPolicy, MultiInputLstmPolicy
    if algorithm == TRAIN_ALGORITHM_MASKABLE_PPO:
        from sb3_contrib.ppo_mask import CnnPolicy, MultiInputPolicy

        return CnnPolicy, MultiInputPolicy
    if algorithm == TRAIN_ALGORITHM_SAC:
        from stable_baselines3.sac import CnnPolicy, MultiInputPolicy

        return CnnPolicy, MultiInputPolicy

    from stable_baselines3.ppo import CnnPolicy, MultiInputPolicy

    return CnnPolicy, MultiInputPolicy


def _full_model_class_for_algorithm(algorithm: str):
    try:
        if algorithm == TRAIN_ALGORITHM_MASKABLE_RECURRENT_PPO:
            from sb3x import MaskableRecurrentPPO

            return MaskableRecurrentPPO
        if algorithm == TRAIN_ALGORITHM_HYBRID_ACTION_PPO:
            from sb3x import HybridActionPPO

            return HybridActionPPO
        if algorithm == TRAIN_ALGORITHM_HYBRID_RECURRENT_PPO:
            from sb3x import HybridRecurrentPPO

            return HybridRecurrentPPO
        if algorithm == TRAIN_ALGORITHM_MASKABLE_HYBRID_ACTION_PPO:
            from sb3x import MaskableHybridActionPPO

            return MaskableHybridActionPPO
        if algorithm == TRAIN_ALGORITHM_MASKABLE_HYBRID_RECURRENT_PPO:
            from sb3x import MaskableHybridRecurrentPPO

            return MaskableHybridRecurrentPPO
    except ImportError as exc:
        raise RuntimeError(
            f"Loading {algorithm} checkpoints requires sb3x in the active environment."
        ) from exc
    raise ValueError(f"Unsupported full-model policy algorithm: {algorithm!r}")


def _load_saved_policy_algorithm(run_dir: Path | None) -> str:
    if run_dir is None:
        return TRAIN_ALGORITHM_PPO

    config_path = run_dir / RUN_LAYOUT.config_filename
    if not config_path.is_file():
        return TRAIN_ALGORITHM_PPO

    from rl_fzerox.core.config import load_train_app_config

    config = load_train_app_config(config_path)
    algorithm = config.train.algorithm
    if algorithm in SAVED_POLICY_ALGORITHMS:
        return algorithm
    if algorithm in LEGACY_PPO_ALGORITHMS:
        # COMPAT SHIM: historical runs may have saved `auto` or plain `ppo`
        # before maskable variants became mandatory; only those explicit values
        # use legacy loading.
        return TRAIN_ALGORITHM_PPO
    raise RuntimeError(f"Unsupported saved policy algorithm: {algorithm!r}")


def _artifact_kind_from_policy_path(policy_path: Path) -> str:
    filename = policy_path.name
    if filename.startswith("latest_"):
        return "latest"
    if filename.startswith("best_"):
        return "best"
    if filename.startswith("final_"):
        return "final"
    raise ValueError(f"Unsupported policy artifact filename: {filename}")
