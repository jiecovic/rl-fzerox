# src/rl_fzerox/core/manager/storage/serialization.py
"""Canonical JSON serialization for manager-owned run-spec snapshots."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, MutableMapping

from rl_fzerox.core.manager.run_spec import ManagedRunConfig


def config_json(config: ManagedRunConfig) -> str:
    """Serialize a managed config in the canonical DB/hash form."""

    return _stable_json(config.model_dump(mode="json"))


def config_hash(config: ManagedRunConfig) -> str:
    """Return a stable short hash for a managed config snapshot."""

    return hashlib.sha256(config_json(config).encode("utf-8")).hexdigest()[:16]


def load_config_json(data: str) -> ManagedRunConfig:
    """Load a managed config snapshot from its canonical JSON form."""

    loaded = json.loads(data)
    if not isinstance(loaded, Mapping):
        raise ValueError("managed run config JSON must decode to an object")
    loaded = _normalize_legacy_config_keys(dict(loaded))
    return ManagedRunConfig.model_validate(loaded)


def _stable_json(data: object) -> str:
    return json.dumps(data, indent=None, separators=(",", ":"), sort_keys=True)


# Temporary compatibility for persisted pre-can-boost-rename snapshots. Remove
# these maps and _normalize_legacy_config_keys after managed config snapshots
# have been migrated during a stopped-run window.
_LEGACY_STRING_VALUES = {
    "vehicle_state.boost_unlocked": "vehicle_state.can_boost",
    "vehicle_state.boost_available": "vehicle_state.can_boost",
}

_LEGACY_REWARD_KEYS = {
    "boost_pad_reward_before_unlock": "boost_pad_reward_cannot_boost",
    "boost_pad_reward_after_unlock": "boost_pad_reward_can_boost",
    "boost_pad_reward_before_can_boost": "boost_pad_reward_cannot_boost",
    "boost_pad_reward_after_can_boost": "boost_pad_reward_can_boost",
}


def _normalize_legacy_config_keys(config: dict[str, object]) -> dict[str, object]:
    _normalize_legacy_strings(config)
    reward = config.get("reward")
    if isinstance(reward, MutableMapping):
        for old_key, new_key in _LEGACY_REWARD_KEYS.items():
            if old_key in reward:
                value = reward.pop(old_key)
                reward.setdefault(new_key, value)
    return config


def _normalize_legacy_strings(value: object) -> object:
    if isinstance(value, str):
        return _LEGACY_STRING_VALUES.get(value, value)
    if isinstance(value, MutableMapping):
        for key, child in tuple(value.items()):
            value[key] = _normalize_legacy_strings(child)
        return value
    if isinstance(value, list):
        for index, child in enumerate(value):
            value[index] = _normalize_legacy_strings(child)
        return value
    return value
