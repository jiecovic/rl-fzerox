# src/rl_fzerox/core/training/session/model/preload.py
from __future__ import annotations

from rl_fzerox.core.config.schema import TrainConfig
from rl_fzerox.core.training.runs import (
    load_train_run_train_config,
    resolve_model_artifact_path,
)


def maybe_preload_training_parameters(*, model, train_config: TrainConfig) -> None:
    """Warm-start a fresh training model from a saved run artifact, if configured.

    We intentionally copy only learned parameters into the freshly built model.
    The current train config remains authoritative for optimizer settings,
    rollout sizes, logging, and output paths.
    """

    if train_config.init_run_dir is None:
        return

    init_run_dir = train_config.init_run_dir.resolve()
    source_train_config = load_train_run_train_config(init_run_dir)
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
