# src/rl_fzerox/core/training/session/model.py
from __future__ import annotations

from rl_fzerox.core.config.schema import PolicyConfig, TrainAppConfig, TrainConfig
from rl_fzerox.core.training.runs import RunPaths


def build_ppo_model(
    *,
    train_env,
    train_config: TrainConfig,
    policy_config: PolicyConfig,
    tensorboard_log: str | None,
):
    """Construct the PPO model configured for the current run."""

    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise RuntimeError(
            "stable-baselines3 and torch are required for training. "
            "Install with `python -m pip install -e .[train]`."
        ) from exc

    from rl_fzerox.core.policy import FZeroXObservationCnnExtractor

    policy_kwargs = {
        "features_extractor_class": FZeroXObservationCnnExtractor,
        "features_extractor_kwargs": {
            "features_dim": policy_config.extractor.features_dim,
        },
        "net_arch": {
            "pi": [int(value) for value in policy_config.net_arch.pi],
            "vf": [int(value) for value in policy_config.net_arch.vf],
        },
        "activation_fn": resolve_policy_activation_fn(policy_config.activation),
    }

    return PPO(
        policy="CnnPolicy",
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
        print(f"run_dir: {run_paths.run_dir}")
        print(f"runtime_root: {run_paths.runtime_root}")
        print(f"device: {model.device}")
        print(f"seed: {config.seed}")
        print(f"observation_space: {train_env.observation_space}")
        print(f"action_space: {train_env.action_space}")
        print(
            "ppo: "
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
