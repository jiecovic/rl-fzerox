# src/rl_fzerox/core/manager/serialization.py
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

from rl_fzerox.core.manager.config import ManagedRunConfig


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
    return ManagedRunConfig.model_validate(loaded)


def _stable_json(data: object) -> str:
    return json.dumps(data, indent=None, separators=(",", ":"), sort_keys=True)
