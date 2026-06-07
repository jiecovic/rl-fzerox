# src/rl_fzerox/core/manager/storage/serialization.py
"""Canonical JSON serialization for manager-owned run-spec snapshots."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

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
    return ManagedRunConfig.model_validate(_with_current_defaults(loaded))


def _stable_json(data: object) -> str:
    return json.dumps(data, indent=None, separators=(",", ":"), sort_keys=True)


def _with_current_defaults(data: Mapping[str, object]) -> dict[str, object]:
    normalized = dict(data)
    action = normalized.get("action")
    if isinstance(action, Mapping):
        normalized_action = dict(action)
        normalized_action.setdefault("hard_zero_ground_pitch", False)
        normalized["action"] = normalized_action
    train = normalized.get("train")
    if isinstance(train, Mapping):
        normalized_train = dict(train)
        actor_regularization = normalized_train.get("actor_regularization")
        if isinstance(actor_regularization, Mapping):
            normalized_actor_regularization = dict(actor_regularization)
        else:
            normalized_actor_regularization = {}
        normalized_actor_regularization.setdefault("pitch_std_cap_loss_weight", 0.0)
        normalized_actor_regularization.setdefault("pitch_std_cap", 0.5)
        normalized_train["actor_regularization"] = normalized_actor_regularization
        normalized["train"] = normalized_train
    return normalized
