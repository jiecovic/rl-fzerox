# src/rl_fzerox/core/training/session/model/preload.py
from __future__ import annotations

from pathlib import Path
from typing import Protocol, TypeVar, cast

from stable_baselines3.common.utils import FloatSchedule

from rl_fzerox.core.runtime_spec.schema import TrainConfig
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
    if train_config.resume_source_algorithm is not None:
        source_algorithm = train_config.resume_source_algorithm
    else:
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
) -> None:
    """Reapply the current run config after full-model resume.

    SB3 full-model load restores the serialized algorithm object exactly,
    including saved hyperparameters. That is correct for optimizer/replay state,
    but it makes YAML changes to future-training knobs such as `gamma` or
    `ent_coef` ineffective. After loading, keep the checkpoint state while
    making the current run config authoritative for the next optimization steps.
    """

    _sync_shared_training_config(model, train_config=train_config)
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
    clip_range_vf = (
        None if train_config.clip_range_vf is None else FloatSchedule(train_config.clip_range_vf)
    )
    _set_attr_if_present(model, "clip_range_vf", clip_range_vf)
    _set_attr_if_present(model, "normalize_advantage", train_config.normalize_advantage)
    _set_attr_if_present(model, "target_kl", train_config.target_kl)
    _set_attr_if_present(model, "stats_window_size", train_config.stats_window_size)
    rollout_buffer = getattr(model, "rollout_buffer", None)
    if rollout_buffer is not None:
        _set_attr_if_present(rollout_buffer, "gamma", train_config.gamma)
        _set_attr_if_present(rollout_buffer, "gae_lambda", train_config.gae_lambda)
def _set_attr_if_present(model: object, attr: str, value: object) -> None:
    if hasattr(model, attr):
        _set_attr(model, attr, value)


def _set_attr(model: object, attr: str, value: object) -> None:
    setattr(model, attr, value)
