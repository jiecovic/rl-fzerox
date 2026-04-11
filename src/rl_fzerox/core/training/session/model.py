# src/rl_fzerox/core/training/session/model.py
from __future__ import annotations

from rl_fzerox.core.config.schema import (
    PolicyConfig,
    TrainAppConfig,
    TrainConfig,
)
from rl_fzerox.core.training.runs import (
    RunPaths,
    load_train_run_config,
    resolve_model_artifact_path,
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
    if effective_algorithm == "sac":
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
    if effective_algorithm == "sac":
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

    _validate_recurrent_configuration_alignment(
        effective_algorithm=effective_algorithm,
        policy_config=policy_config,
    )
    _validate_masking_configuration(
        train_env=train_env,
        effective_algorithm=effective_algorithm,
    )

    algorithm_class = _resolve_training_algorithm(effective_algorithm)

    from gymnasium import spaces

    recurrent_enabled = policy_config.recurrent.enabled
    if isinstance(train_env.observation_space, spaces.Dict):
        policy_name = (
            "MultiInputLstmPolicy" if recurrent_enabled else "MultiInputPolicy"
        )
    else:
        policy_name = "CnnLstmPolicy" if recurrent_enabled else "CnnPolicy"

    policy_kwargs = _policy_kwargs(
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
    policy_name = (
        "MultiInputPolicy" if isinstance(train_env.observation_space, spaces.Dict) else "CnnPolicy"
    )
    policy_kwargs = _policy_kwargs(
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


def maybe_preload_training_parameters(*, model, train_config: TrainConfig) -> None:
    """Warm-start a fresh training model from a saved run artifact, if configured.

    We intentionally copy only learned parameters into the freshly built model.
    The current train config remains authoritative for optimizer settings,
    rollout sizes, logging, and output paths.
    """

    if train_config.init_run_dir is None:
        return

    init_run_dir = train_config.init_run_dir.resolve()
    source_train_config = load_train_run_config(init_run_dir).train
    if source_train_config.algorithm != train_config.algorithm:
        raise RuntimeError(
            "Warm-start checkpoint algorithm mismatch: "
            f"source={source_train_config.algorithm}, current={train_config.algorithm}. "
            "Use a checkpoint produced by the same training algorithm."
        )

    model_path = resolve_model_artifact_path(
        init_run_dir,
        artifact=train_config.init_artifact,
    )
    model.set_parameters(str(model_path), exact_match=True, device=model.device)


def validate_training_algorithm_config(config: TrainAppConfig) -> None:
    """Reject incompatible algorithm/config combinations before training starts."""

    if config.train.algorithm == "ppo":
        raise RuntimeError(
            "Plain PPO training is no longer supported. "
            "Use `train.algorithm=maskable_ppo` or `maskable_recurrent_ppo`."
        )
    if config.train.algorithm != "sac":
        return
    if config.env.action.name != "continuous_steer_drive":
        raise RuntimeError(
            "SAC requires env.action.name=continuous_steer_drive so the action space is Box"
        )
    if config.env.action.mask is not None:
        raise RuntimeError("SAC does not support env.action.mask; use the continuous adapter")
    if config.curriculum.enabled:
        raise RuntimeError("SAC does not support action-mask curriculum stages")
    if config.env.observation.mode == "image_state" and config.train.optimize_memory_usage:
        raise RuntimeError(
            "SAC optimize_memory_usage is not supported with Dict image_state observations"
        )


def training_requires_action_masks(config: TrainAppConfig) -> bool:
    """Return whether the current env stack depends on action masking.

    PPO-family training always relies on mask-aware algorithms because gameplay
    masks are part of that discrete-action contract. SAC uses a separate
    continuous action adapter without boost/drift mask branches.
    """

    return config.train.algorithm != "sac"


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


def _resolve_training_algorithm(algorithm: str):
    try:
        if algorithm == "maskable_recurrent_ppo":
            from sb3x import MaskableRecurrentPPO

            return MaskableRecurrentPPO
        if algorithm == "maskable_ppo":
            from sb3_contrib import MaskablePPO

            return MaskablePPO

        from stable_baselines3 import PPO

        return PPO
    except ImportError as exc:
        if algorithm == "maskable_recurrent_ppo":
            raise RuntimeError(
                "Maskable recurrent PPO requires stable-baselines3, sb3-contrib, "
                "torch, and sb3x. Install train deps and then install sb3x in "
                "the active environment."
            ) from exc
        raise RuntimeError(
            "stable-baselines3 and torch are required for training. "
            "Install with `python -m pip install -e .[train]`."
        ) from exc


def _policy_kwargs(
    *,
    train_env,
    policy_config: PolicyConfig,
    value_head_key: str,
) -> dict[str, object]:
    from gymnasium import spaces

    from rl_fzerox.core.policy import FZeroXImageStateExtractor, FZeroXObservationCnnExtractor

    if isinstance(train_env.observation_space, spaces.Dict):
        extractor_class = FZeroXImageStateExtractor
        extractor_kwargs = {
            "features_dim": policy_config.extractor.features_dim,
            "state_features_dim": policy_config.extractor.state_features_dim,
        }
    else:
        extractor_class = FZeroXObservationCnnExtractor
        extractor_kwargs = {
            "features_dim": policy_config.extractor.features_dim,
        }

    return {
        "features_extractor_class": extractor_class,
        "features_extractor_kwargs": extractor_kwargs,
        "net_arch": {
            "pi": [int(value) for value in policy_config.net_arch.pi],
            value_head_key: [int(value) for value in policy_config.net_arch.vf],
        },
        "activation_fn": resolve_policy_activation_fn(policy_config.activation),
    }


def _validate_masking_configuration(*, train_env, effective_algorithm: str) -> None:
    if effective_algorithm not in {"maskable_ppo", "maskable_recurrent_ppo"}:
        return

    if not hasattr(train_env, "env_method"):
        raise RuntimeError("Maskable PPO requires a vector env exposing env_method()")
    if not train_env.has_attr("action_masks"):
        raise RuntimeError("Maskable PPO requires env.action_masks() support")


def _validate_recurrent_configuration_alignment(
    *,
    effective_algorithm: str,
    policy_config: PolicyConfig,
) -> None:
    recurrent_enabled = policy_config.recurrent.enabled
    if recurrent_enabled and effective_algorithm != "maskable_recurrent_ppo":
        raise RuntimeError(
            "Recurrent policy config requires train.algorithm=maskable_recurrent_ppo"
        )
    if not recurrent_enabled and effective_algorithm == "maskable_recurrent_ppo":
        raise RuntimeError(
            "maskable_recurrent_ppo requires policy.recurrent.enabled=true"
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
    if effective_algorithm == "sac":
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
