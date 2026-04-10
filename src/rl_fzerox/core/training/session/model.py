# src/rl_fzerox/core/training/session/model.py
from __future__ import annotations

from rl_fzerox.core.config.schema import (
    PolicyConfig,
    TrainAppConfig,
    TrainConfig,
)
from rl_fzerox.core.training.runs import RunPaths


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
    _validate_masking_configuration(
        train_env=train_env,
        effective_algorithm=effective_algorithm,
    )

    algorithm_class = _resolve_training_algorithm(effective_algorithm)

    from gymnasium import spaces

    from rl_fzerox.core.policy import FZeroXImageStateExtractor, FZeroXObservationCnnExtractor

    if isinstance(train_env.observation_space, spaces.Dict):
        policy_name = "MultiInputPolicy"
        extractor_class = FZeroXImageStateExtractor
        extractor_kwargs = {
            "features_dim": policy_config.extractor.features_dim,
            "state_features_dim": policy_config.extractor.state_features_dim,
        }
    else:
        policy_name = "CnnPolicy"
        extractor_class = FZeroXObservationCnnExtractor
        extractor_kwargs = {
            "features_dim": policy_config.extractor.features_dim,
        }

    policy_kwargs = {
        "features_extractor_class": extractor_class,
        "features_extractor_kwargs": extractor_kwargs,
        "net_arch": {
            "pi": [int(value) for value in policy_config.net_arch.pi],
            "vf": [int(value) for value in policy_config.net_arch.vf],
        },
        "activation_fn": resolve_policy_activation_fn(policy_config.activation),
    }

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
        ent_coef=train_config.ent_coef,
        vf_coef=train_config.vf_coef,
        max_grad_norm=train_config.max_grad_norm,
        policy_kwargs=policy_kwargs,
        tensorboard_log=tensorboard_log,
        verbose=train_config.verbose,
        device=train_config.device,
    )


def validate_training_algorithm_config(config: TrainAppConfig) -> None:
    """Reject incompatible algorithm/config combinations before training starts."""

    if config.train.algorithm == "ppo":
        raise RuntimeError(
            "Plain PPO training is no longer supported. "
            "Use `train.algorithm=maskable_ppo` or omit the field."
        )


def training_requires_action_masks(_config: TrainAppConfig) -> bool:
    """Return whether the current env stack depends on action masking.

    Training always relies on MaskablePPO now because gameplay masks are part of
    the base env contract, not only optional curriculum/static configuration.
    """

    return True


def resolve_effective_training_algorithm(
    *,
    train_config: TrainConfig,
    masking_required: bool,
) -> str:
    """Resolve the configured train.algorithm into the concrete algorithm used.

    `auto` is now a backwards-compatible alias for `maskable_ppo`. Plain PPO is
    only retained as a legacy value so older saved run configs still load.
    """

    _ = masking_required
    if train_config.algorithm == "auto":
        return "maskable_ppo"
    return train_config.algorithm


def _resolve_training_algorithm(algorithm: str):
    try:
        if algorithm == "maskable_ppo":
            from sb3_contrib import MaskablePPO

            return MaskablePPO

        from stable_baselines3 import PPO

        return PPO
    except ImportError as exc:
        raise RuntimeError(
            "stable-baselines3 and torch are required for training. "
            "Install with `python -m pip install -e .[train]`."
        ) from exc


def _validate_masking_configuration(*, train_env, effective_algorithm: str) -> None:
    if effective_algorithm != "maskable_ppo":
        return

    if not hasattr(train_env, "env_method"):
        raise RuntimeError("Maskable PPO requires a vector env exposing env_method()")
    if not train_env.has_attr("action_masks"):
        raise RuntimeError("Maskable PPO requires env.action_masks() support")


def resolve_policy_activation_fn(name: str):
    """Map the configured SB3 policy-head activation name to a torch module."""

    from torch import nn

    if name == "tanh":
        return nn.Tanh
    if name == "relu":
        return nn.ReLU
    raise ValueError(f"Unsupported policy activation: {name!r}")


def build_tensorboard_logger(run_paths: RunPaths):
    """Create the SB3 TensorBoard logger for one training run."""

    try:
        from stable_baselines3.common import logger as sb3_logger
    except ImportError as exc:
        raise RuntimeError(
            "stable-baselines3 is required for training. "
            "Install with `python -m pip install -e .[train]`."
        ) from exc

    return sb3_logger.configure(str(run_paths.tensorboard_dir), ["tensorboard"])


def print_training_startup(
    *,
    model,
    train_env,
    config: TrainAppConfig,
    run_paths: RunPaths,
) -> None:
    """Print one compact startup summary for the current train run."""

    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
    except ImportError:
        effective_algorithm = resolve_effective_training_algorithm(
            train_config=config.train,
            masking_required=training_requires_action_masks(config),
        )
        print(f"run_dir: {run_paths.run_dir}")
        print(f"runtime_root: {run_paths.runtime_root}")
        print(f"device: {model.device}")
        print(f"seed: {config.seed}")
        print(f"observation_space: {train_env.observation_space}")
        print(f"action_space: {train_env.action_space}")
        print(
            "ppo: "
            f"algo={effective_algorithm} "
            f"vec_env={config.train.vec_env} "
            f"num_envs={config.train.num_envs} "
            f"total_timesteps={config.train.total_timesteps} "
            f"n_steps={config.train.n_steps} "
            f"batch_size={config.train.batch_size} "
            f"lr={config.train.learning_rate}"
        )
        print(model.policy)
        return

    console = Console()
    effective_algorithm = resolve_effective_training_algorithm(
        train_config=config.train,
        masking_required=training_requires_action_masks(config),
    )
    summary = Table(show_header=False, box=None, pad_edge=False)
    summary.add_column(style="bold cyan")
    summary.add_column()
    summary.add_row("run_dir", str(run_paths.run_dir))
    summary.add_row("runtime_root", str(run_paths.runtime_root))
    summary.add_row("device", str(model.device))
    summary.add_row("seed", str(config.seed))
    summary.add_row("observation", str(train_env.observation_space))
    summary.add_row("action", str(train_env.action_space))
    summary.add_row(
        "ppo",
        " ".join(
            [
                f"algo={effective_algorithm}",
                f"vec_env={config.train.vec_env}",
                f"num_envs={config.train.num_envs}",
                f"total_timesteps={config.train.total_timesteps}",
                f"n_steps={config.train.n_steps}",
                f"batch_size={config.train.batch_size}",
                f"lr={config.train.learning_rate}",
            ]
        ),
    )
    console.print(Panel(summary, title="Training", expand=False))
    console.print(Panel(str(model.policy), title="Policy", expand=False))
