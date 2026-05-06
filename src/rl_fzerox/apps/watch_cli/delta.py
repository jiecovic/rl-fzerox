# src/rl_fzerox/apps/watch_cli/delta.py
from __future__ import annotations

from collections.abc import Mapping, Sequence

from omegaconf import OmegaConf

from rl_fzerox.core.runtime_spec.schema import WatchAppConfig


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
    if not key:
        return None
    if key.startswith("hydra."):
        raise ValueError("Hydra overrides are no longer supported")
    if key.startswith("~"):
        raise ValueError("Deletion overrides are not supported for watch sessions")
    if key.startswith("/") or "@" in key:
        raise ValueError("Config-group overrides are not supported for watch sessions")
    if not separator:
        raise ValueError(f"Watch override must use key=value syntax: {override!r}")
    return f"{key}={value}"


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


def string_key_mapping(value: object) -> dict[str, object] | None:
    if not isinstance(value, dict):
        return None

    mapping: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            return None
        mapping[key] = item
    return mapping
