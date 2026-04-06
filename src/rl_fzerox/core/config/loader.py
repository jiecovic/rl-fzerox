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
from rl_fzerox.core.config.paths import config_root_dir, resolve_config_path_value


def load_watch_app_config(
    config_path: Path,
    overrides: Sequence[str] | None = None,
) -> WatchAppConfig:
    """Compose a watch config with Hydra and validate it with Pydantic."""

    resolved_config_path = config_path.expanduser().resolve()
    try:
        config_data = _compose_config_data(
            config_path=resolved_config_path,
            overrides=overrides,
        )
        _resolve_emulator_paths(config_data, config_dir=resolved_config_path.parent)
        return WatchAppConfig.model_validate(config_data)
    except (HydraException, OmegaConfBaseException, OSError, TypeError, ValidationError) as exc:
        raise ValueError(
            f"Could not load watch config from {resolved_config_path}: {exc}"
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


def _resolve_emulator_paths(config_data: dict[str, object], *, config_dir: Path) -> None:
    emulator = config_data.get("emulator")
    if not isinstance(emulator, dict):
        return

    for field_name in (
        "core_path",
        "rom_path",
        "runtime_dir",
        "baseline_state_path",
    ):
        raw_value = emulator.get(field_name)
        if not isinstance(raw_value, str):
            continue

        emulator[field_name] = str(
            resolve_config_path_value(raw_value, config_dir=config_dir)
        )
