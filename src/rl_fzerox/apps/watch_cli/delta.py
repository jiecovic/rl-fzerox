# src/rl_fzerox/apps/watch_cli/delta.py
from __future__ import annotations

from collections.abc import Mapping, Sequence

from omegaconf import OmegaConf

from rl_fzerox.core.config.schema import WatchAppConfig


def watch_config_delta(
    base_config: WatchAppConfig,
    overridden_config: WatchAppConfig,
    overrides: Sequence[str],
) -> dict[str, object]:
    """Return only the effective CLI changes after Hydra composition."""

    delta = mapping_delta(
        watch_config_data(base_config),
        watch_config_data(overridden_config),
    )
    overridden_data = watch_config_data(overridden_config)
    for override in overrides:
        key_path = override_key_path(override)
        if key_path is None:
            continue
        override_value = mapping_path_value(overridden_data, key_path)
        if override_value is None:
            continue
        value, path = override_value
        set_mapping_path_value(delta, path, value)
    return delta


def watch_config_delta_from_dotlist(overrides: Sequence[str]) -> dict[str, object]:
    """Parse direct CLI overrides without composing an external watch YAML."""

    dotlist = [
        dotlist_override
        for override in overrides
        if (dotlist_override := direct_dotlist_override(override)) is not None
    ]
    if not dotlist:
        return {}
    delta = string_key_mapping(
        OmegaConf.to_container(OmegaConf.from_dotlist(dotlist), resolve=True)
    )
    if delta is None:
        raise ValueError("Watch overrides must resolve to a string-keyed mapping")
    return delta


def apply_watch_config_delta(
    config: WatchAppConfig,
    delta: Mapping[str, object],
) -> WatchAppConfig:
    """Apply CLI override delta after run-manifest inheritance."""

    if not delta:
        return config
    return WatchAppConfig.model_validate(
        merge_mapping(
            watch_config_data(config),
            delta,
        )
    )


def direct_dotlist_override(override: str) -> str | None:
    key, separator, value = override.partition("=")
    key = key.strip()
    while key.startswith("+"):
        key = key[1:]
    if not key or key.startswith("hydra."):
        return None
    if key.startswith("~"):
        raise ValueError("Deletion overrides require --config")
    if key.startswith("/") or "@" in key:
        raise ValueError("Config-group overrides require --config")
    if not separator:
        raise ValueError(f"Watch override must use key=value syntax: {override!r}")
    return f"{key}={value}"


def mapping_delta(
    base: Mapping[str, object],
    overridden: Mapping[str, object],
) -> dict[str, object]:
    delta: dict[str, object] = {}
    for key, overridden_value in overridden.items():
        if key not in base:
            delta[key] = overridden_value
            continue

        base_value = base[key]
        base_mapping = string_key_mapping(base_value)
        overridden_mapping = string_key_mapping(overridden_value)
        if base_mapping is not None and overridden_mapping is not None:
            nested_delta = mapping_delta(base_mapping, overridden_mapping)
            if nested_delta:
                delta[key] = nested_delta
            continue

        if overridden_value != base_value:
            delta[key] = overridden_value
    return delta


def merge_mapping(
    base: Mapping[str, object],
    update: Mapping[str, object],
) -> dict[str, object]:
    merged: dict[str, object] = dict(base)
    for key, update_value in update.items():
        existing_mapping = string_key_mapping(merged.get(key))
        update_mapping = string_key_mapping(update_value)
        if existing_mapping is not None and update_mapping is not None:
            merged[key] = merge_mapping(existing_mapping, update_mapping)
            continue
        merged[key] = update_value
    return merged


def watch_config_data(config: WatchAppConfig) -> dict[str, object]:
    data = string_key_mapping(config.model_dump(mode="json", exclude_none=False))
    if data is None:
        raise TypeError("Watch config dump must be a string-keyed mapping")
    return data


def override_key_path(override: str) -> tuple[str, ...] | None:
    key = override.split("=", maxsplit=1)[0].strip()
    while key.startswith("+"):
        key = key[1:]
    if key.startswith("~"):
        key = key[1:]
    if not key or key.startswith("hydra."):
        return None
    if key.startswith("/"):
        key = key.rsplit("/", maxsplit=1)[-1]
    if "@" in key:
        key = key.split("@", maxsplit=1)[0]

    path = tuple(part for part in key.split(".") if part)
    return path or None


def mapping_path_value(
    mapping: Mapping[str, object],
    path: tuple[str, ...],
) -> tuple[object, tuple[str, ...]] | None:
    current: object = mapping
    for part in path:
        current_mapping = string_key_mapping(current)
        if current_mapping is None or part not in current_mapping:
            return None
        current = current_mapping[part]
    return current, path


def set_mapping_path_value(
    mapping: dict[str, object],
    path: tuple[str, ...],
    value: object,
) -> None:
    current = mapping
    for part in path[:-1]:
        next_mapping = string_key_mapping(current.get(part))
        if next_mapping is None:
            next_mapping = {}
            current[part] = next_mapping
        current = next_mapping
    current[path[-1]] = value


def string_key_mapping(value: object) -> dict[str, object] | None:
    if not isinstance(value, dict):
        return None

    mapping: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            return None
        mapping[key] = item
    return mapping

