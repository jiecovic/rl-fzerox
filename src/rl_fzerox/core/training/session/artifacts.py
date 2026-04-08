# src/rl_fzerox/core/training/session/artifacts.py
from __future__ import annotations

import os
import shutil
from pathlib import Path

from rl_fzerox.core.config.schema import TrainAppConfig
from rl_fzerox.core.training.runs import RunPaths, materialize_train_run_config


def _resolve_train_run_config(
    *,
    config: TrainAppConfig,
    run_paths: RunPaths,
) -> TrainAppConfig:
    """Resolve one train config snapshot with a run-local runtime root."""

    return materialize_train_run_config(config, run_paths=run_paths)


def _validate_training_baseline_state(config: TrainAppConfig) -> None:
    """Fail clearly when a configured local training baseline is missing."""

    baseline_state_path = config.emulator.baseline_state_path
    if baseline_state_path is None:
        return
    if baseline_state_path.exists():
        return
    raise RuntimeError(
        "Configured training baseline state does not exist: "
        f"{baseline_state_path}. Create it from watch with "
        "`emulator.baseline_state_path` set and press `K` at race start."
    )


def _save_latest_artifacts(model, run_paths: RunPaths) -> None:
    _save_artifacts_atomically(
        model=model,
        model_path=run_paths.latest_model_path,
        policy_path=run_paths.latest_policy_path,
    )


def _save_artifacts_atomically(*, model, model_path: Path, policy_path: Path) -> None:
    _atomic_save_artifact(model.save, model_path)
    _atomic_save_artifact(model.policy.save, policy_path)


def _atomic_save_artifact(save_fn, target_path: Path) -> None:
    tmp_path = target_path.with_name(f".{target_path.stem}.tmp{target_path.suffix}")
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        save_fn(str(tmp_path))
        os.replace(tmp_path, target_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _cleanup_failed_run(run_paths: RunPaths, model: object | None) -> None:
    if not run_paths.run_dir.exists():
        return

    num_timesteps = getattr(model, "num_timesteps", None) if model is not None else None
    if num_timesteps not in (None, 0):
        return

    shutil.rmtree(run_paths.run_dir, ignore_errors=True)
