# src/rl_fzerox/core/config/loader.py
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TypeVar

from hydra import compose, initialize_config_dir
from hydra.errors import HydraException
from omegaconf import OmegaConf
from omegaconf.errors import OmegaConfBaseException
from pydantic import ValidationError

from rl_fzerox.core.config.paths import config_root_dir, resolve_config_path_value
from rl_fzerox.core.config.schema import TrainAppConfig, WatchAppConfig

ConfigModel = TypeVar("ConfigModel", WatchAppConfig, TrainAppConfig)


def load_watch_app_config(
    config_path: Path,
    overrides: Sequence[str] | None = None,
) -> WatchAppConfig:
    """Compose a watch config with Hydra and validate it with Pydantic."""

    return _load_app_config(
        config_path=config_path,
        overrides=overrides,
        config_kind="watch",
        model_type=WatchAppConfig,
        path_fields={
            "emulator": (
                "core_path",
                "rom_path",
                "runtime_dir",
                "baseline_state_path",
            ),
            "watch": ("policy_run_dir",),
        },
    )


def load_train_app_config(
    config_path: Path,
    overrides: Sequence[str] | None = None,
) -> TrainAppConfig:
    """Compose a train config with Hydra and validate it with Pydantic."""

    return _load_app_config(
        config_path=config_path,
        overrides=overrides,
        config_kind="train",
        model_type=TrainAppConfig,
        path_fields={
            "emulator": (
                "core_path",
                "rom_path",
                "runtime_dir",
                "baseline_state_path",
            ),
            "train": (
                "output_root",
                "init_run_dir",
            ),
        },
    )


def _load_app_config(
    *,
    config_path: Path,
    overrides: Sequence[str] | None,
    config_kind: str,
    model_type: type[ConfigModel],
    path_fields: dict[str, tuple[str, ...]],
) -> ConfigModel:
    resolved_config_path = config_path.expanduser().resolve()
    try:
        config_data = _compose_config_data(
            config_path=resolved_config_path,
            overrides=overrides,
        )
        _resolve_config_paths(
            config_data,
            config_dir=resolved_config_path.parent,
            path_fields=path_fields,
        )
        return model_type.model_validate(config_data)
    except (HydraException, OmegaConfBaseException, OSError, TypeError, ValidationError) as exc:
        raise ValueError(
            f"Could not load {config_kind} config from {resolved_config_path}: {exc}"
        ) from exc


def _compose_config_data(
    *,
    config_path: Path,
    overrides: Sequence[str] | None,
) -> dict[str, object]:
    config_dir = config_path.parent
    if not config_dir.is_dir():
        raise FileNotFoundError(f"Config directory not found: {config_dir}")

    hydra_overrides = list(overrides or [])
    config_root = config_root_dir().resolve()
    if config_dir.resolve() != config_root:
        hydra_overrides = [
            f"hydra.searchpath=[file://{config_root.as_posix()}]",
            *hydra_overrides,
        ]

    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        composed = compose(config_name=config_path.stem, overrides=hydra_overrides)

    loaded = OmegaConf.to_container(composed, resolve=True)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise TypeError("Config must resolve to a mapping")

    normalized: dict[str, object] = {}
    for key, value in loaded.items():
        if not isinstance(key, str):
            raise TypeError("Config keys must be strings")
        normalized[key] = value
    normalized.pop("hydra", None)
    return normalized


def _resolve_config_paths(
    config_data: dict[str, object],
    *,
    config_dir: Path,
    path_fields: dict[str, tuple[str, ...]],
) -> None:
    for section_name, field_names in path_fields.items():
        section = config_data.get(section_name)
        if not isinstance(section, dict):
            continue

        for field_name in field_names:
            raw_value = section.get(field_name)
            if not isinstance(raw_value, str):
                continue

            section[field_name] = str(resolve_config_path_value(raw_value, config_dir=config_dir))
