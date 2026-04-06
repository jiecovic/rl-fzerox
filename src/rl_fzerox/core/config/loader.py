# src/rl_fzerox/core/config/loader.py
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.errors import HydraException
from omegaconf import OmegaConf
from omegaconf.errors import OmegaConfBaseException
from pydantic import ValidationError

from rl_fzerox.core.config.models import WatchAppConfig


def default_config_dir() -> Path:
    return Path(__file__).resolve().parents[4] / "conf"


def load_watch_app_config(
    *,
    config_name: str = "watch",
    overrides: Sequence[str] | None = None,
    config_dir: Path | None = None,
) -> WatchAppConfig:
    resolved_config_dir = (config_dir or default_config_dir()).resolve()
    try:
        config_data = _compose_config_data(
            config_name=config_name,
            overrides=overrides,
            config_dir=resolved_config_dir,
        )
        return WatchAppConfig.model_validate(config_data)
    except (HydraException, OmegaConfBaseException, ValidationError) as exc:
        raise ValueError(
            f"Could not load config '{config_name}' from {resolved_config_dir}: {exc}"
        ) from exc


def _compose_config_data(
    *,
    config_name: str,
    overrides: Sequence[str] | None = None,
    config_dir: Path | None = None,
) -> dict[str, object]:
    resolved_config_dir = (config_dir or default_config_dir()).resolve()
    with initialize_config_dir(
        config_dir=str(resolved_config_dir),
        version_base=None,
    ):
        composed = compose(config_name=config_name, overrides=list(overrides or []))

    data = OmegaConf.to_container(composed, resolve=True)
    if not isinstance(data, dict):
        raise TypeError("Config must resolve to a mapping")

    normalized: dict[str, object] = {}
    for key, value in data.items():
        if not isinstance(key, str):
            raise TypeError("Config keys must be strings")
        normalized[key] = value
    normalized.pop("hydra", None)
    return normalized
