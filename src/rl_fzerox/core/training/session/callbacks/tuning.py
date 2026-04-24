# src/rl_fzerox/core/training/session/callbacks/tuning.py
from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

from rl_fzerox.core.config.schema import CurriculumTrainOverridesConfig

from .metrics import CallbackLogger


@runtime_checkable
class PpoTunableModel(Protocol):
    learning_rate: float | Callable[[float], float]
    lr_schedule: Callable[[float], float]
    clip_range: Callable[[float], float]
    n_epochs: int
    batch_size: int
    ent_coef: float
    policy: object


@runtime_checkable
class _OptimizerLike(Protocol):
    param_groups: list[dict[str, object]]


@runtime_checkable
class _PolicyWithOptimizer(Protocol):
    optimizer: _OptimizerLike


def apply_stage_train_overrides(
    *,
    model: object,
    overrides: CurriculumTrainOverridesConfig | None,
) -> None:
    if overrides is None:
        return
    if not isinstance(model, PpoTunableModel):
        raise RuntimeError("Curriculum train overrides require a PPO-family model")
    if overrides.learning_rate is not None:
        _set_model_learning_rate(model, float(overrides.learning_rate))
    if overrides.n_epochs is not None:
        model.n_epochs = int(overrides.n_epochs)
    if overrides.batch_size is not None:
        model.batch_size = int(overrides.batch_size)
    if overrides.clip_range is not None:
        model.clip_range = constant_schedule(float(overrides.clip_range))
    if overrides.ent_coef is not None:
        model.ent_coef = float(overrides.ent_coef)


def record_stage_train_overrides(
    *,
    logger: CallbackLogger,
    overrides: CurriculumTrainOverridesConfig | None,
) -> None:
    if overrides is None:
        return
    if overrides.learning_rate is not None:
        logger.record("curriculum/learning_rate", float(overrides.learning_rate))
    if overrides.n_epochs is not None:
        logger.record("curriculum/n_epochs", int(overrides.n_epochs))
    if overrides.batch_size is not None:
        logger.record("curriculum/batch_size", int(overrides.batch_size))
    if overrides.clip_range is not None:
        logger.record("curriculum/clip_range", float(overrides.clip_range))
    if overrides.ent_coef is not None:
        logger.record("curriculum/ent_coef", float(overrides.ent_coef))


def constant_schedule(value: float) -> Callable[[float], float]:
    def schedule(_progress_remaining: float) -> float:
        return value

    return schedule


def _set_model_learning_rate(model: PpoTunableModel, learning_rate: float) -> None:
    model.learning_rate = learning_rate
    model.lr_schedule = constant_schedule(learning_rate)
    policy = model.policy
    if not isinstance(policy, _PolicyWithOptimizer):
        return
    for param_group in policy.optimizer.param_groups:
        param_group["lr"] = learning_rate
