# src/rl_fzerox/core/training/session/model/startup.py
from __future__ import annotations

from rl_fzerox.core.config.schema import TrainAppConfig, TrainConfig
from rl_fzerox.core.domain.training_algorithms import TRAIN_ALGORITHM_SAC
from rl_fzerox.core.training.runs import RunPaths
from rl_fzerox.core.training.session.model.algorithms import resolve_effective_training_algorithm


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
        _print_plain_startup(
            model=model,
            train_env=train_env,
            config=config,
            run_paths=run_paths,
        )
        return

    console = Console()
    effective_algorithm = resolve_effective_training_algorithm(
        train_config=config.train,
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
        "train",
        " ".join(
            _training_summary_parts(
                train_config=config.train,
                effective_algorithm=effective_algorithm,
            )
        ),
    )
    if config.policy.recurrent.enabled:
        summary.add_row(
            "lstm",
            " ".join(
                [
                    f"hidden={config.policy.recurrent.hidden_size}",
                    f"layers={config.policy.recurrent.n_lstm_layers}",
                    f"shared={config.policy.recurrent.shared_lstm}",
                    f"critic={config.policy.recurrent.enable_critic_lstm}",
                ]
            ),
        )
    console.print(Panel(summary, title="Training", expand=False))
    console.print(Panel(str(model.policy), title="Policy", expand=False))


def _print_plain_startup(
    *,
    model,
    train_env,
    config: TrainAppConfig,
    run_paths: RunPaths,
) -> None:
    effective_algorithm = resolve_effective_training_algorithm(
        train_config=config.train,
    )
    print(f"run_dir: {run_paths.run_dir}")
    print(f"runtime_root: {run_paths.runtime_root}")
    print(f"device: {model.device}")
    print(f"seed: {config.seed}")
    print(f"observation_space: {train_env.observation_space}")
    print(f"action_space: {train_env.action_space}")
    print(
        "train: "
        + " ".join(
            _training_summary_parts(
                train_config=config.train,
                effective_algorithm=effective_algorithm,
            )
        )
    )
    if config.policy.recurrent.enabled:
        print(
            "lstm: "
            f"hidden={config.policy.recurrent.hidden_size} "
            f"layers={config.policy.recurrent.n_lstm_layers} "
            f"shared={config.policy.recurrent.shared_lstm} "
            f"critic={config.policy.recurrent.enable_critic_lstm}"
        )
    print(model.policy)


def _training_summary_parts(
    *,
    train_config: TrainConfig,
    effective_algorithm: str,
) -> list[str]:
    parts = [
        f"algo={effective_algorithm}",
        f"vec_env={train_config.vec_env}",
        f"num_envs={train_config.num_envs}",
        f"total_timesteps={train_config.total_timesteps}",
        f"batch_size={train_config.batch_size}",
        f"lr={train_config.learning_rate}",
    ]
    if effective_algorithm == TRAIN_ALGORITHM_SAC:
        parts.extend(
            [
                f"buffer_size={train_config.buffer_size}",
                f"learning_starts={train_config.learning_starts}",
                f"train_freq={train_config.train_freq}",
                f"gradient_steps={train_config.gradient_steps}",
            ]
        )
        return parts
    parts.append(f"n_steps={train_config.n_steps}")
    return parts
