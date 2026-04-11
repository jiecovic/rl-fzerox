# src/rl_fzerox/core/training/session/model/algorithms.py
from __future__ import annotations

from rl_fzerox.core.config.schema import TrainAppConfig, TrainConfig

MASKABLE_TRAINING_ALGORITHMS = {
    "auto",
    "maskable_ppo",
    "maskable_recurrent_ppo",
    "maskable_hybrid_action_ppo",
    "maskable_hybrid_recurrent_ppo",
}
RECURRENT_TRAINING_ALGORITHMS = {
    "maskable_recurrent_ppo",
    "hybrid_recurrent_ppo",
    "maskable_hybrid_recurrent_ppo",
}
SB3X_TRAINING_ALGORITHMS = {
    "maskable_recurrent_ppo",
    "hybrid_action_ppo",
    "hybrid_recurrent_ppo",
    "maskable_hybrid_action_ppo",
    "maskable_hybrid_recurrent_ppo",
}


def training_requires_action_masks(config: TrainAppConfig) -> bool:
    """Return whether the current env stack depends on action masking."""

    return config.train.algorithm in MASKABLE_TRAINING_ALGORITHMS


def resolve_effective_training_algorithm(
    *,
    train_config: TrainConfig,
    masking_required: bool,
) -> str:
    """Resolve the configured train.algorithm into the concrete algorithm used.

    `auto` is now a backwards-compatible alias for `maskable_ppo`. Recurrent
    training must be selected explicitly so the saved run config is unambiguous.
    """

    _ = masking_required
    if train_config.algorithm == "auto":
        return "maskable_ppo"
    return train_config.algorithm


def resolve_training_algorithm_class(algorithm: str):
    try:
        if algorithm == "maskable_recurrent_ppo":
            from sb3x import MaskableRecurrentPPO

            return MaskableRecurrentPPO
        if algorithm == "maskable_ppo":
            from sb3_contrib import MaskablePPO

            return MaskablePPO
        if algorithm == "hybrid_action_ppo":
            from sb3x import HybridActionPPO

            return HybridActionPPO
        if algorithm == "hybrid_recurrent_ppo":
            from sb3x import HybridRecurrentPPO

            return HybridRecurrentPPO
        if algorithm == "maskable_hybrid_action_ppo":
            from sb3x import MaskableHybridActionPPO

            return MaskableHybridActionPPO
        if algorithm == "maskable_hybrid_recurrent_ppo":
            from sb3x import MaskableHybridRecurrentPPO

            return MaskableHybridRecurrentPPO

        from stable_baselines3 import PPO

        return PPO
    except ImportError as exc:
        if algorithm in SB3X_TRAINING_ALGORITHMS:
            raise RuntimeError(
                f"{algorithm} requires stable-baselines3, torch, and sb3x. "
                "Install train deps and then install sb3x in "
                "the active environment."
            ) from exc
        raise RuntimeError(
            "stable-baselines3 and torch are required for training. "
            "Install with `python -m pip install -e .[train]`."
        ) from exc
