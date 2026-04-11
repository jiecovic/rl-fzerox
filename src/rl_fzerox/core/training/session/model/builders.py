# src/rl_fzerox/core/training/session/model/builders.py
from __future__ import annotations

from rl_fzerox.core.config.schema import PolicyConfig, TrainConfig
from rl_fzerox.core.domain.training_algorithms import TRAIN_ALGORITHM_SAC
from rl_fzerox.core.training.session.model.algorithms import (
    resolve_effective_training_algorithm,
    resolve_training_algorithm_class,
)
from rl_fzerox.core.training.session.model.policy import (
    build_policy_kwargs,
    resolve_policy_name,
)
from rl_fzerox.core.training.session.model.validation import (
    validate_masking_configuration,
    validate_recurrent_configuration_alignment,
)


def build_training_model(
    *,
    train_env,
    train_config: TrainConfig,
    policy_config: PolicyConfig,
    tensorboard_log: str | None,
    masking_required: bool,
):
    """Construct the configured SB3 model for the current run."""

    effective_algorithm = resolve_effective_training_algorithm(
        train_config=train_config,
        masking_required=masking_required,
    )
    if effective_algorithm == TRAIN_ALGORITHM_SAC:
        return _build_sac_model(
            train_env=train_env,
            train_config=train_config,
            policy_config=policy_config,
            tensorboard_log=tensorboard_log,
        )
    return _build_ppo_family_model(
        train_env=train_env,
        train_config=train_config,
        policy_config=policy_config,
        tensorboard_log=tensorboard_log,
        effective_algorithm=effective_algorithm,
    )


def build_ppo_model(
    *,
    train_env,
    train_config: TrainConfig,
    policy_config: PolicyConfig,
    tensorboard_log: str | None,
    masking_required: bool,
):
    """Construct the configured PPO-family model for the current run."""

    effective_algorithm = resolve_effective_training_algorithm(
        train_config=train_config,
        masking_required=masking_required,
    )
    if effective_algorithm == TRAIN_ALGORITHM_SAC:
        raise RuntimeError("build_ppo_model cannot construct SAC; use build_training_model")
    return _build_ppo_family_model(
        train_env=train_env,
        train_config=train_config,
        policy_config=policy_config,
        tensorboard_log=tensorboard_log,
        effective_algorithm=effective_algorithm,
    )


def _build_ppo_family_model(
    *,
    train_env,
    train_config: TrainConfig,
    policy_config: PolicyConfig,
    tensorboard_log: str | None,
    effective_algorithm: str,
):
    """Construct a PPO-family model for the current run."""

    validate_recurrent_configuration_alignment(
        effective_algorithm=effective_algorithm,
        policy_config=policy_config,
    )
    validate_masking_configuration(
        train_env=train_env,
        effective_algorithm=effective_algorithm,
    )

    algorithm_class = resolve_training_algorithm_class(effective_algorithm)
    recurrent_enabled = policy_config.recurrent.enabled
    policy_name = resolve_policy_name(
        train_env=train_env,
        recurrent_enabled=recurrent_enabled,
    )
    policy_kwargs = build_policy_kwargs(
        train_env=train_env,
        policy_config=policy_config,
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

    return algorithm_class(
        policy=policy_name,
        env=train_env,
        learning_rate=train_config.learning_rate,
        n_steps=train_config.n_steps,
        n_epochs=train_config.n_epochs,
        batch_size=train_config.batch_size,
        gamma=train_config.gamma,
        gae_lambda=train_config.gae_lambda,
        clip_range=train_config.clip_range,
        ent_coef=_ppo_ent_coef(train_config),
        vf_coef=train_config.vf_coef,
        max_grad_norm=train_config.max_grad_norm,
        policy_kwargs=policy_kwargs,
        tensorboard_log=tensorboard_log,
        verbose=train_config.verbose,
        device=train_config.device,
    )


def _build_sac_model(
    *,
    train_env,
    train_config: TrainConfig,
    policy_config: PolicyConfig,
    tensorboard_log: str | None,
):
    """Construct a SAC model for the continuous steering experiment."""

    from gymnasium import spaces
    from stable_baselines3 import SAC

    if not isinstance(train_env.action_space, spaces.Box):
        raise RuntimeError("SAC requires a continuous Box action space")
    policy_name = resolve_policy_name(train_env=train_env, recurrent_enabled=False)
    policy_kwargs = build_policy_kwargs(
        train_env=train_env,
        policy_config=policy_config,
        value_head_key="qf",
    )
    return SAC(
        policy=policy_name,
        env=train_env,
        learning_rate=train_config.learning_rate,
        buffer_size=train_config.buffer_size,
        learning_starts=train_config.learning_starts,
        batch_size=train_config.batch_size,
        tau=train_config.tau,
        gamma=train_config.gamma,
        train_freq=train_config.train_freq,
        gradient_steps=train_config.gradient_steps,
        ent_coef=train_config.ent_coef,
        target_update_interval=train_config.target_update_interval,
        target_entropy=train_config.target_entropy,
        optimize_memory_usage=train_config.optimize_memory_usage,
        policy_kwargs=policy_kwargs,
        tensorboard_log=tensorboard_log,
        verbose=train_config.verbose,
        device=train_config.device,
    )


def _ppo_ent_coef(train_config: TrainConfig) -> float:
    ent_coef = train_config.ent_coef
    if ent_coef == "auto":
        raise RuntimeError("PPO-family algorithms require a numeric train.ent_coef")
    return float(ent_coef)
