# src/rl_fzerox/core/runtime_spec/inference.py
"""Inference-facing runtime config cleanup shared by eval and Career playback."""

from __future__ import annotations

from rl_fzerox.core.runtime_spec.schema import TrainAppConfig


def inference_train_app_config(config: TrainAppConfig) -> TrainAppConfig:
    """Return a runtime config with train-only episode randomization disabled.

    Evaluation and Career playback should preserve the checkpoint's action and
    observation contract, but they must not replay train-time regularizers.
    Probabilities in ``(0, 1)`` are stochastic regularization and become off.
    ``p == 1`` is a deterministic contract: that action branch or state feature
    was always unavailable during training, so inference must keep it unavailable.
    """

    action_config = config.env.action.model_copy(
        update={
            "lean_episode_mask_probability": _deterministic_episode_dropout_probability(
                config.env.action.lean_episode_mask_probability
            ),
            "air_brake_episode_mask_probability": _deterministic_episode_dropout_probability(
                config.env.action.air_brake_episode_mask_probability
            ),
            "spin_episode_mask_probability": _deterministic_episode_dropout_probability(
                config.env.action.spin_episode_mask_probability
            ),
        }
    )
    train_config = config.train.model_copy(
        update={
            "state_feature_dropout_groups": tuple(
                group.model_copy(update={"dropout_prob": 1.0})
                for group in config.train.state_feature_dropout_groups
                if group.dropout_prob >= 1.0
            )
        }
    )
    return config.model_copy(
        update={
            "env": config.env.model_copy(update={"action": action_config}),
            "train": train_config,
        }
    )


def _deterministic_episode_dropout_probability(probability: float) -> float:
    return 1.0 if probability >= 1.0 else 0.0
