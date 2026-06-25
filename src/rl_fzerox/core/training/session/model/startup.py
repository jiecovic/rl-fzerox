# src/rl_fzerox/core/training/session/model/startup.py
"""TensorBoard logger setup and human-readable training startup summaries."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import prod
from typing import TYPE_CHECKING

import torch as th
from gymnasium import spaces

from rl_fzerox.core.runtime_spec.schema import TrainAppConfig, TrainConfig
from rl_fzerox.core.training.runs import RunPaths
from rl_fzerox.core.training.session.model.algorithms import resolve_effective_training_algorithm

if TYPE_CHECKING:
    from stable_baselines3.common.vec_env import VecEnv


def build_tensorboard_logger(run_paths: RunPaths, *, step_offset: int = 0):
    """Create the SB3 TensorBoard logger for one training run."""

    try:
        from stable_baselines3.common import logger as sb3_logger
    except ImportError as exc:
        raise RuntimeError(
            "stable-baselines3 is required for training. "
            "Install with `python -m pip install -e .[train]`."
        ) from exc

    logger = sb3_logger.configure(str(run_paths.tensorboard_dir), ["tensorboard"])
    if step_offset <= 0:
        return logger

    class TensorboardStepOffsetLogger(sb3_logger.Logger):
        """Apply one fixed global-step offset before SB3 writes TensorBoard events."""

        def __init__(self, *, step_offset: int) -> None:
            super().__init__(folder=logger.dir, output_formats=logger.output_formats)
            self._step_offset = step_offset

        def dump(self, step: int = 0) -> None:
            super().dump(step + self._step_offset)

    return TensorboardStepOffsetLogger(step_offset=step_offset)


def print_training_startup(
    *,
    model: object,
    train_env: VecEnv,
    config: TrainAppConfig,
    run_paths: RunPaths,
) -> None:
    """Print one compact startup summary for the current train run."""

    startup_summary = _training_startup_summary(
        model=model,
        train_env=train_env,
        config=config,
        run_paths=run_paths,
    )
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
    except ImportError:
        _print_plain_startup(startup_summary)
        return

    console = Console()
    summary = Table(show_header=False, box=None, pad_edge=False)
    summary.add_column(style="bold cyan")
    summary.add_column()
    summary.add_row("run_dir", startup_summary.run_dir)
    summary.add_row("runtime_root", startup_summary.runtime_root)
    summary.add_row("device", startup_summary.device)
    summary.add_row("seed", startup_summary.seed)
    summary.add_row("observation", startup_summary.observation_space)
    summary.add_row("action", startup_summary.action_space)
    summary.add_row("train", startup_summary.train)
    if startup_summary.lstm is not None:
        summary.add_row("lstm", startup_summary.lstm)
    console.print(Panel(summary, title="Training", expand=False))
    console.print(
        Panel(
            _memory_summary_table(startup_summary.memory),
            title="Memory",
            expand=False,
        )
    )
    console.print(Panel(startup_summary.policy, title="Policy", expand=False))


@dataclass(frozen=True, slots=True)
class _TrainingStartupSummary:
    run_dir: str
    runtime_root: str
    device: str
    seed: str
    observation_space: str
    action_space: str
    train: str
    lstm: str | None
    memory: _TrainingMemorySummary
    policy: str


def _training_startup_summary(
    *,
    model: object,
    train_env: VecEnv,
    config: TrainAppConfig,
    run_paths: RunPaths,
) -> _TrainingStartupSummary:
    effective_algorithm = resolve_effective_training_algorithm(
        train_config=config.train,
    )
    memory_summary = _training_memory_summary(
        model=model,
        observation_space=train_env.observation_space,
        batch_size=config.train.batch_size,
    )
    train = " ".join(
        _training_summary_parts(
            train_config=config.train,
            effective_algorithm=effective_algorithm,
        )
    )
    lstm = None
    if config.policy.recurrent.enabled:
        lstm = " ".join(
            [
                f"hidden={config.policy.recurrent.hidden_size}",
                f"layers={config.policy.recurrent.n_lstm_layers}",
                f"shared={config.policy.recurrent.shared_lstm}",
                f"critic={config.policy.recurrent.enable_critic_lstm}",
            ]
        )
    return _TrainingStartupSummary(
        run_dir=str(run_paths.run_dir),
        runtime_root=str(run_paths.runtime_root),
        device=str(_model_device(model)),
        seed=str(config.seed),
        observation_space=str(train_env.observation_space),
        action_space=str(train_env.action_space),
        train=train,
        lstm=lstm,
        memory=memory_summary,
        policy=_model_policy_text(model),
    )


def _print_plain_startup(summary: _TrainingStartupSummary) -> None:
    print(f"run_dir: {summary.run_dir}")
    print(f"runtime_root: {summary.runtime_root}")
    print(f"device: {summary.device}")
    print(f"seed: {summary.seed}")
    print(f"observation_space: {summary.observation_space}")
    print(f"action_space: {summary.action_space}")
    print(f"train: {summary.train}")
    if summary.lstm is not None:
        print(f"lstm: {summary.lstm}")
    print("memory:")
    print(f"  params: {summary.memory.parameter_count}")
    if summary.memory.cuda_now is not None:
        print(f"  cuda_now: {summary.memory.cuda_now}")
    if summary.memory.cuda_estimate is not None:
        print(f"  cuda_est: {summary.memory.cuda_estimate}")
    print(summary.policy)


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
    if train_config.resume_run_dir is not None:
        parts.append(f"resume={train_config.resume_mode}:{train_config.resume_artifact}")
    parts.append(f"n_steps={train_config.n_steps}")
    return parts


@dataclass(frozen=True, slots=True)
class _TrainingMemorySummary:
    parameter_count: str
    cuda_now: str | None
    cuda_estimate: str | None


def _memory_summary_table(memory_summary: _TrainingMemorySummary):
    from rich.table import Table

    summary = Table(show_header=False, box=None, pad_edge=False)
    summary.add_column(style="bold cyan")
    summary.add_column()
    summary.add_row("params", memory_summary.parameter_count)
    if memory_summary.cuda_now is not None:
        summary.add_row("cuda_now", memory_summary.cuda_now)
    if memory_summary.cuda_estimate is not None:
        summary.add_row("cuda_est", memory_summary.cuda_estimate)
    return summary


@dataclass(frozen=True, slots=True)
class _ModelParameterSummary:
    total_count: int
    trainable_count: int
    total_bytes: int
    trainable_bytes: int


def _training_memory_summary(
    *,
    model: object,
    observation_space: spaces.Space[object],
    batch_size: int,
) -> _TrainingMemorySummary:
    """Estimate startup memory pressure without allocating a rollout buffer."""

    device = _model_device(model)
    parameter_summary = _model_parameter_summary(model)
    minibatch_observation_bytes = _minibatch_observation_bytes(
        observation_space=observation_space,
        batch_size=batch_size,
    )
    cuda_estimate = None
    if _device_type(device) == "cuda":
        estimated_total = (
            parameter_summary.total_bytes
            + parameter_summary.trainable_bytes
            + (2 * parameter_summary.trainable_bytes)
            + minibatch_observation_bytes
        )
        cuda_estimate = " ".join(
            [
                f"params={_format_bytes(parameter_summary.total_bytes)}",
                f"grads~={_format_bytes(parameter_summary.trainable_bytes)}",
                f"adam~={_format_bytes(2 * parameter_summary.trainable_bytes)}",
                f"batch_obs~={_format_bytes(minibatch_observation_bytes)}",
                f"total~={_format_bytes(estimated_total)}",
            ]
        )
    return _TrainingMemorySummary(
        parameter_count=_format_parameter_summary(parameter_summary),
        cuda_now=_cuda_now_summary(device),
        cuda_estimate=cuda_estimate,
    )


def _cuda_now_summary(device: object) -> str | None:
    device_text = str(device)
    if _device_type(device) != "cuda":
        return None
    allocated = th.cuda.memory_allocated(device_text)
    reserved = th.cuda.memory_reserved(device_text)
    free, total = th.cuda.mem_get_info(device_text)
    return " ".join(
        [
            f"alloc={_format_bytes(allocated)}",
            f"reserved={_format_bytes(reserved)}",
            f"free={_format_bytes(free)}",
            f"total={_format_bytes(total)}",
        ]
    )


def _model_parameter_summary(model: object) -> _ModelParameterSummary:
    parameters = _iter_model_parameters(model)
    total_count = 0
    trainable_count = 0
    total_bytes = 0
    trainable_bytes = 0
    for parameter in parameters:
        parameter_count = int(parameter.numel())
        parameter_bytes = parameter_count * int(parameter.element_size())
        total_count += parameter_count
        total_bytes += parameter_bytes
        if parameter.requires_grad:
            trainable_count += parameter_count
            trainable_bytes += parameter_bytes
    return _ModelParameterSummary(
        total_count=total_count,
        trainable_count=trainable_count,
        total_bytes=total_bytes,
        trainable_bytes=trainable_bytes,
    )


def _iter_model_parameters(model: object) -> tuple[th.nn.Parameter, ...]:
    policy = _model_policy(model)
    policy_parameters = _parameter_iterable(policy)
    if policy_parameters is not None:
        return policy_parameters
    model_parameters = _parameter_iterable(model)
    if model_parameters is not None:
        return model_parameters
    return ()


def _parameter_iterable(candidate: object) -> tuple[th.nn.Parameter, ...] | None:
    parameters = getattr(candidate, "parameters", None)
    if not callable(parameters):
        return None
    raw_parameters = parameters()
    if not isinstance(raw_parameters, Iterable):
        return None
    return tuple(
        parameter for parameter in raw_parameters if isinstance(parameter, th.nn.Parameter)
    )


def _model_device(model: object) -> object:
    return getattr(model, "device", "cpu")


def _model_policy(model: object) -> object | None:
    return getattr(model, "policy", None)


def _model_policy_text(model: object) -> str:
    policy = _model_policy(model)
    if policy is not None:
        return str(policy)
    return str(model)


def _minibatch_observation_bytes(
    *,
    observation_space: spaces.Space[object],
    batch_size: int,
) -> int:
    if isinstance(observation_space, spaces.Dict):
        return sum(
            _minibatch_box_bytes(space, batch_size=batch_size)
            for space in observation_space.spaces.values()
            if isinstance(space, spaces.Box)
        )
    if isinstance(observation_space, spaces.Box):
        return _minibatch_box_bytes(observation_space, batch_size=batch_size)
    return 0


def _minibatch_box_bytes(space: spaces.Box, *, batch_size: int) -> int:
    """Estimate float32 tensor storage for one SB3 minibatch observation."""

    value_count = max(1, int(batch_size)) * prod(int(dim) for dim in space.shape)
    tensor_element_size = 4
    return value_count * tensor_element_size


def _device_type(device: object) -> str:
    raw_type = getattr(device, "type", None)
    if isinstance(raw_type, str):
        return raw_type
    return str(device).split(":", 1)[0]


def _format_bytes(byte_count: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    value = float(max(0, byte_count))
    unit_index = 0
    while value >= 1024.0 and unit_index < len(units) - 1:
        value /= 1024.0
        unit_index += 1
    if unit_index == 0:
        return f"{int(value)}{units[unit_index]}"
    return f"{value:.1f}{units[unit_index]}"


def _format_parameter_summary(summary: _ModelParameterSummary) -> str:
    total = _format_parameter_count(summary.total_count)
    trainable = _format_parameter_count(summary.trainable_count)
    if summary.total_count == summary.trainable_count:
        return total
    return f"total={total} trainable={trainable}"


def _format_parameter_count(parameter_count: int) -> str:
    units = (
        (1_000_000_000, "B"),
        (1_000_000, "M"),
        (1_000, "K"),
    )
    value = max(0, int(parameter_count))
    for threshold, suffix in units:
        if value >= threshold:
            return f"{value / threshold:.1f}{suffix}"
    return str(value)
