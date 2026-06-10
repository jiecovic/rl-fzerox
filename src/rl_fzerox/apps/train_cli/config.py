# src/rl_fzerox/apps/train_cli/config.py
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from omegaconf import OmegaConf

from rl_fzerox.core.runtime_spec.schema import TrainAppConfig


def continue_saved_run_config(
    config: TrainAppConfig,
    *,
    continue_run_dir: Path,
    continue_artifact: str,
    overrides: Sequence[str],
) -> TrainAppConfig:
    """Apply optional dotlist overrides to a saved config, then force in-place resume."""

    data = config.model_dump(mode="json", exclude_unset=True)
    if overrides:
        merged = OmegaConf.merge(OmegaConf.create(data), OmegaConf.from_dotlist(list(overrides)))
        loaded = OmegaConf.to_container(merged, resolve=True)
        if not isinstance(loaded, dict):
            raise ValueError("Saved-run overrides must resolve to a mapping")
        data = loaded

    train_data = data.setdefault("train", {})
    if not isinstance(train_data, dict):
        raise ValueError("Saved train config must contain a train mapping")
    train_data.update(
        {
            "continue_run_dir": continue_run_dir,
            "resume_run_dir": continue_run_dir,
            "resume_artifact": continue_artifact,
            "resume_mode": "full_model",
        }
    )
    return TrainAppConfig.model_validate(data)
