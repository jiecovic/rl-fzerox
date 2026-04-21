# src/rl_fzerox/core/training/session/model/preload.py
from __future__ import annotations

from pathlib import Path
from typing import Protocol, TypeVar, cast

from rl_fzerox.core.config.schema import TrainConfig
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
        return algorithm_class.load(
            str(model_path),
            env=train_env,
            device=train_config.device,
        )

    _set_model_parameters(model, model_path)
    return model


def _set_model_parameters(model: object, model_path: Path) -> None:
    set_parameters = getattr(model, "set_parameters", None)
    if not callable(set_parameters):
        raise TypeError("Training model does not expose set_parameters(...)")
    device = getattr(model, "device", "auto")
    set_parameters(str(model_path), exact_match=True, device=device)
