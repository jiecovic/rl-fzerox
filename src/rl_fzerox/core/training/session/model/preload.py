# src/rl_fzerox/core/training/session/model/preload.py
from __future__ import annotations

import math
from pathlib import Path
from typing import Protocol, TypeVar, cast

import numpy as np
import torch as th
from stable_baselines3.common.utils import FloatSchedule

from rl_fzerox.core.config.schema import TrainConfig
from rl_fzerox.core.domain.training_algorithms import TRAINING_ALGORITHMS
from rl_fzerox.core.training.runs import (
    load_train_run_train_config,
    resolve_model_artifact_path,
)
from rl_fzerox.core.training.session.model.algorithms import (
    resolve_effective_training_algorithm,
    resolve_training_algorithm_class,
)

_ModelT = TypeVar("_ModelT")
_ModelT_co = TypeVar("_ModelT_co", covariant=True)


class _FullModelLoader(Protocol[_ModelT_co]):
    @classmethod
    def load(cls, path: str, *, env: object, device: str) -> _ModelT_co: ...


class _ProgressSchedule(Protocol):
    def __call__(self, progress_remaining: float) -> float: ...


def maybe_resume_training_model(
    *,
    model: _ModelT,
    train_env: object,
    train_config: TrainConfig,
) -> _ModelT:
    """Resume or warm-start a training model from a saved run artifact.

    `weights_only` keeps the current config authoritative and copies learned
    parameters into the freshly built model. `full_model` loads optimizer and
    scheduler state too, then attaches the current env.
    """

    if train_config.resume_run_dir is None:
        return model

    resume_run_dir = train_config.resume_run_dir.resolve()
    source_train_config = load_train_run_train_config(resume_run_dir)
    source_algorithm = resolve_effective_training_algorithm(train_config=source_train_config)
    current_algorithm = resolve_effective_training_algorithm(train_config=train_config)
    if source_algorithm != current_algorithm:
        raise RuntimeError(
            "Resume checkpoint algorithm mismatch: "
            f"source={source_algorithm}, current={current_algorithm}. "
            "Use a checkpoint produced by the same training algorithm."
        )

    model_path = resolve_model_artifact_path(
        resume_run_dir,
        artifact=train_config.resume_artifact,
    )
    if train_config.resume_mode == "full_model":
        # The algorithm check above guarantees the artifact and current model
        # are from the same SB3 family; the class factory itself is dynamic.
        algorithm_class = cast(
            _FullModelLoader[_ModelT],
            resolve_training_algorithm_class(current_algorithm),
        )
        loaded_model = algorithm_class.load(
            str(model_path),
            env=train_env,
            device=train_config.device,
        )
        _sync_loaded_model_training_config(
            loaded_model,
            train_config=train_config,
            algorithm=current_algorithm,
        )
        return loaded_model

    _set_model_parameters(model, model_path)
    return model


def _set_model_parameters(model: object, model_path: Path) -> None:
    set_parameters = getattr(model, "set_parameters", None)
    if not callable(set_parameters):
        raise TypeError("Training model does not expose set_parameters(...)")
    device = getattr(model, "device", "auto")
    set_parameters(str(model_path), exact_match=True, device=device)


def _sync_loaded_model_training_config(
    model: object,
    *,
    train_config: TrainConfig,
    algorithm: str,
) -> None:
    """Reapply the current run config after full-model resume.

    SB3 full-model load restores the serialized algorithm object exactly,
    including saved hyperparameters. That is correct for optimizer/replay state,
    but it makes YAML changes to future-training knobs such as `gamma` or
    `ent_coef` ineffective. After loading, keep the checkpoint state while
    making the current run config authoritative for the next optimization steps.
    """

    _sync_shared_training_config(model, train_config=train_config)
    if algorithm in TRAINING_ALGORITHMS.sac_family:
        _sync_loaded_sac_training_config(model, train_config=train_config)
        return
    _sync_loaded_ppo_training_config(model, train_config=train_config)


def _sync_shared_training_config(model: object, *, train_config: TrainConfig) -> None:
    _set_attr_if_present(model, "learning_rate", train_config.learning_rate)
    setup_lr_schedule = getattr(model, "_setup_lr_schedule", None)
    if callable(setup_lr_schedule):
        setup_lr_schedule()
    _set_attr_if_present(model, "verbose", train_config.verbose)


def _sync_loaded_ppo_training_config(model: object, *, train_config: TrainConfig) -> None:
    _set_attr_if_present(model, "batch_size", train_config.batch_size)
    _set_attr_if_present(model, "n_epochs", train_config.n_epochs)
    _set_attr_if_present(model, "gamma", train_config.gamma)
    _set_attr_if_present(model, "gae_lambda", train_config.gae_lambda)
    _set_attr_if_present(model, "ent_coef", float(train_config.ent_coef))
    _set_attr_if_present(model, "vf_coef", train_config.vf_coef)
    _set_attr_if_present(model, "max_grad_norm", train_config.max_grad_norm)
    _set_attr_if_present(model, "clip_range", FloatSchedule(train_config.clip_range))
    rollout_buffer = getattr(model, "rollout_buffer", None)
    if rollout_buffer is not None:
        _set_attr_if_present(rollout_buffer, "gamma", train_config.gamma)
        _set_attr_if_present(rollout_buffer, "gae_lambda", train_config.gae_lambda)


def _sync_loaded_sac_training_config(model: object, *, train_config: TrainConfig) -> None:
    _set_attr_if_present(model, "batch_size", train_config.batch_size)
    _set_attr_if_present(model, "gamma", train_config.gamma)
    _set_attr_if_present(model, "learning_starts", train_config.learning_starts)
    _set_attr_if_present(model, "tau", train_config.tau)
    _set_attr_if_present(model, "gradient_steps", train_config.gradient_steps)
    _set_attr_if_present(model, "target_update_interval", train_config.target_update_interval)
    if hasattr(model, "train_freq"):
        _set_attr(model, "train_freq", train_config.train_freq)
        convert_train_freq = getattr(model, "_convert_train_freq", None)
        if callable(convert_train_freq):
            convert_train_freq()

    if hasattr(model, "target_entropy"):
        _set_attr(
            model,
            "target_entropy",
            _resolved_sac_target_entropy(
                model,
                configured_target_entropy=train_config.target_entropy,
            ),
        )
    _sync_sac_entropy_coefficient(model, ent_coef=train_config.ent_coef)


def _sync_sac_entropy_coefficient(model: object, *, ent_coef: float | str) -> None:
    if ent_coef == "auto":
        current_tensor = _effective_ent_coef_tensor(model)
        initial_value = float(current_tensor.detach().item()) if current_tensor is not None else 1.0
        if initial_value <= 0.0:
            initial_value = 1.0
        device = getattr(model, "device", "cpu")
        _set_attr(model, "ent_coef", "auto")
        _set_attr(model, "ent_coef_tensor", None)
        log_ent_coef = th.log(th.ones(1, device=device) * initial_value).requires_grad_(True)
        _set_attr(model, "log_ent_coef", log_ent_coef)
        lr_schedule = _require_progress_schedule(model)
        _set_attr(
            model,
            "ent_coef_optimizer",
            th.optim.Adam([log_ent_coef], lr=lr_schedule(1.0)),
        )
        return

    _set_attr(model, "ent_coef", float(ent_coef))
    _set_attr(model, "log_ent_coef", None)
    _set_attr(model, "ent_coef_optimizer", None)
    device = getattr(model, "device", "cpu")
    _set_attr(model, "ent_coef_tensor", th.tensor(float(ent_coef), device=device))


def _set_attr_if_present(model: object, attr: str, value: object) -> None:
    if hasattr(model, attr):
        _set_attr(model, attr, value)


def _set_attr(model: object, attr: str, value: object) -> None:
    setattr(model, attr, value)


def _require_progress_schedule(model: object) -> _ProgressSchedule:
    raw_schedule = getattr(model, "lr_schedule", None)
    if callable(raw_schedule):

        def _schedule(progress_remaining: float) -> float:
            value = raw_schedule(progress_remaining)
            if isinstance(value, (int, float)):
                return float(value)
            raise TypeError("SAC lr_schedule must return a float")

        return _schedule
    raise TypeError("SAC model does not expose lr_schedule after resume")


def _effective_ent_coef_tensor(model: object) -> th.Tensor | None:
    log_ent_coef = getattr(model, "log_ent_coef", None)
    if isinstance(log_ent_coef, th.Tensor):
        return th.exp(log_ent_coef.detach())
    ent_coef_tensor = getattr(model, "ent_coef_tensor", None)
    if isinstance(ent_coef_tensor, th.Tensor):
        return ent_coef_tensor.detach()
    return None


def _resolved_sac_target_entropy(
    model: object,
    *,
    configured_target_entropy: float | str,
) -> float:
    if configured_target_entropy != "auto":
        return float(configured_target_entropy)

    hybrid_action_spec = getattr(model, "hybrid_action_spec", None)
    if hybrid_action_spec is not None:
        discrete_entropy = sum(
            math.log(int(action_dim)) for action_dim in hybrid_action_spec.discrete_action_dims
        )
        return -float(hybrid_action_spec.continuous_dim + discrete_entropy)

    env = getattr(model, "env", None)
    action_space = getattr(env, "action_space", None)
    action_shape = getattr(action_space, "shape", None)
    if action_shape is None:
        raise TypeError("SAC auto target entropy requires an action space shape")
    return float(-np.prod(action_shape).astype(np.float32))
