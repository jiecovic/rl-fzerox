# src/rl_fzerox/core/training/session/model/builders.py
from __future__ import annotations

import torch as th
from stable_baselines3.common.vec_env import VecEnv

from rl_fzerox.core.domain.training_algorithms import TRAINING_ALGORITHMS
from rl_fzerox.core.runtime_spec.schema import EnvConfig, PolicyConfig, TrainConfig
from rl_fzerox.core.training.session.model.action_bias import apply_initial_action_biases
from rl_fzerox.core.training.session.model.algorithms import (
    resolve_effective_training_algorithm,
    resolve_ppo_training_algorithm_class,
)
from rl_fzerox.core.training.session.model.policy import (
    build_policy_kwargs,
    resolve_policy_entry,
)
from rl_fzerox.core.training.session.model.validation import (
    validate_auxiliary_state_configuration,
    validate_masking_configuration,
    validate_recurrent_configuration_alignment,
)


def build_training_model(
    *,
    train_env: VecEnv,
    train_config: TrainConfig,
    policy_config: PolicyConfig,
    env_config: EnvConfig | None = None,
    tensorboard_log: str | None,
):
    """Construct the configured SB3 model for the current run."""

    effective_algorithm = resolve_effective_training_algorithm(
        train_config=train_config,
    )
    return _build_ppo_family_model(
        train_env=train_env,
        train_config=train_config,
        policy_config=policy_config,
        env_config=env_config or EnvConfig(),
        tensorboard_log=tensorboard_log,
        effective_algorithm=effective_algorithm,
    )


def build_ppo_model(
    *,
    train_env: VecEnv,
    train_config: TrainConfig,
    policy_config: PolicyConfig,
    env_config: EnvConfig | None = None,
    tensorboard_log: str | None,
):
    """Construct the configured PPO-family model for the current run."""

    effective_algorithm = resolve_effective_training_algorithm(
        train_config=train_config,
    )
    return _build_ppo_family_model(
        train_env=train_env,
        train_config=train_config,
        policy_config=policy_config,
        env_config=env_config or EnvConfig(),
        tensorboard_log=tensorboard_log,
        effective_algorithm=effective_algorithm,
    )


def _build_ppo_family_model(
    *,
    train_env: VecEnv,
    train_config: TrainConfig,
    policy_config: PolicyConfig,
    env_config: EnvConfig,
    tensorboard_log: str | None,
    effective_algorithm: str,
):
    """Construct a PPO-family model for the current run."""

    _validate_requested_device(train_config.device)
    validate_recurrent_configuration_alignment(
        effective_algorithm=effective_algorithm,
        policy_config=policy_config,
    )
    validate_masking_configuration(
        train_env=train_env,
        effective_algorithm=effective_algorithm,
    )
    validate_auxiliary_state_configuration(
        train_env=train_env,
        policy_config=policy_config,
        effective_algorithm=effective_algorithm,
    )

    algorithm_class = resolve_ppo_training_algorithm_class(effective_algorithm)
    recurrent_enabled = policy_config.recurrent.enabled
    policy_entry = resolve_policy_entry(
        train_env=train_env,
        effective_algorithm=effective_algorithm,
        policy_config=policy_config,
        train_config=train_config,
        recurrent_enabled=recurrent_enabled,
    )
    policy_kwargs = build_policy_kwargs(
        train_env=train_env,
        policy_config=policy_config,
        train_config=train_config,
        env_config=env_config,
        value_head_key="vf",
    )
    if recurrent_enabled:
        policy_kwargs.update(
            {
                "lstm_hidden_size": policy_config.recurrent.hidden_size,
                "n_lstm_layers": policy_config.recurrent.n_lstm_layers,
                "shared_lstm": policy_config.recurrent.shared_lstm,
                "enable_critic_lstm": policy_config.recurrent.enable_critic_lstm,
            }
        )

    model_kwargs: dict[str, object] = {
        "policy": policy_entry,
        "env": train_env,
        "learning_rate": train_config.learning_rate,
        "n_steps": train_config.n_steps,
        "n_epochs": train_config.n_epochs,
        "batch_size": train_config.batch_size,
        "gamma": train_config.gamma,
        "gae_lambda": train_config.gae_lambda,
        "clip_range": train_config.clip_range,
        "clip_range_vf": train_config.clip_range_vf,
        "normalize_advantage": train_config.normalize_advantage,
        "ent_coef": _ppo_ent_coef(train_config),
        "vf_coef": train_config.vf_coef,
        "max_grad_norm": train_config.max_grad_norm,
        "target_kl": train_config.target_kl,
        "stats_window_size": train_config.stats_window_size,
        "policy_kwargs": policy_kwargs,
        "tensorboard_log": tensorboard_log,
        "verbose": train_config.verbose,
        "device": train_config.device,
    }
    if effective_algorithm in TRAINING_ALGORITHMS.hybrid:
        model_kwargs["entropy_group_weights"] = dict(train_config.entropy_group_weights)

    model = algorithm_class(**model_kwargs)
    if train_config.resume_run_dir is None:
        apply_initial_action_biases(
            model,
            train_env=train_env,
            policy_config=policy_config,
        )
    return model


def _ppo_ent_coef(train_config: TrainConfig) -> float:
    return float(train_config.ent_coef)


def _validate_requested_device(device: str) -> None:
    requested = device.lower()
    if requested == "cuda" or requested.startswith("cuda:"):
        if not th.cuda.is_available():
            raise RuntimeError(
                "train.device is set to cuda, but PyTorch cannot access CUDA. "
                "Fix GPU/driver/WSL CUDA access or set train.device=cpu explicitly."
            )
