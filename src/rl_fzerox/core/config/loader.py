# src/rl_fzerox/core/config/loader.py
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError

from rl_fzerox.core.config.models import WatchAppConfig


def load_watch_app_config(config_path: Path) -> WatchAppConfig:
    """Load and validate a watch configuration file."""

    resolved_config_path = config_path.expanduser().resolve()
    try:
        config_data = _load_yaml_mapping(resolved_config_path)
        return WatchAppConfig.model_validate(config_data)
    except (OSError, TypeError, ValidationError, yaml.YAMLError) as exc:
        raise ValueError(
            f"Could not load watch config from {resolved_config_path}: {exc}"
        ) from exc


def _load_yaml_mapping(config_path: Path) -> dict[str, object]:
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)

    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise TypeError("Config must resolve to a mapping")

    normalized: dict[str, object] = {}
    for key, value in loaded.items():
        if not isinstance(key, str):
            raise TypeError("Config keys must be strings")
        normalized[key] = value
    return normalized
