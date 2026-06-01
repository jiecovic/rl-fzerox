# src/rl_fzerox/core/manager/db/repositories/configs.py
"""Repository operations for immutable config snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from rl_fzerox.core.manager.db.models.configs import ConfigSnapshotModel
from rl_fzerox.core.manager.run_spec import ManagedRunConfig
from rl_fzerox.core.manager.storage.serialization import config_hash, config_json, load_config_json

ConfigSnapshotKind = Literal["run", "draft", "template", "import"]
CONFIG_SNAPSHOT_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class ConfigSnapshot:
    """One immutable stored config snapshot."""

    id: str
    kind: ConfigSnapshotKind
    schema_version: int
    created_at: str
    config: ManagedRunConfig
    config_json: str
    config_hash: str


def create_config_snapshot(
    session: Session,
    *,
    kind: ConfigSnapshotKind,
    config: ManagedRunConfig,
    created_at: str,
    snapshot_id: str | None = None,
) -> ConfigSnapshot:
    """Persist and return a canonical validated config snapshot."""

    snapshot_json = config_json(config)
    snapshot_hash = config_hash(config)
    resolved_snapshot_id = snapshot_id or f"cfg_{uuid4().hex}"
    existing = session.get(ConfigSnapshotModel, resolved_snapshot_id)
    if existing is not None:
        if (
            existing.kind != kind
            or existing.schema_version != CONFIG_SNAPSHOT_SCHEMA_VERSION
            or existing.config_json != snapshot_json
            or existing.config_hash != snapshot_hash
        ):
            raise ValueError(
                f"config snapshot id already exists with different content: {snapshot_id}"
            )
        return _snapshot_from_model(existing)
    model = ConfigSnapshotModel(
        id=resolved_snapshot_id,
        kind=kind,
        schema_version=CONFIG_SNAPSHOT_SCHEMA_VERSION,
        created_at=created_at,
        config_json=snapshot_json,
        config_hash=snapshot_hash,
    )
    session.add(model)
    session.flush()
    return ConfigSnapshot(
        id=resolved_snapshot_id,
        kind=kind,
        schema_version=CONFIG_SNAPSHOT_SCHEMA_VERSION,
        created_at=created_at,
        config=config,
        config_json=snapshot_json,
        config_hash=snapshot_hash,
    )


def get_config_snapshot(session: Session, snapshot_id: str) -> ConfigSnapshot | None:
    """Load one config snapshot by id."""

    model = session.get(ConfigSnapshotModel, snapshot_id)
    return None if model is None else _snapshot_from_model(model)


def list_config_snapshots(session: Session) -> tuple[ConfigSnapshot, ...]:
    """Return all config snapshots in stable creation order."""

    models = session.scalars(select(ConfigSnapshotModel).order_by(ConfigSnapshotModel.created_at))
    return tuple(_snapshot_from_model(model) for model in models)


def _snapshot_from_model(model: ConfigSnapshotModel) -> ConfigSnapshot:
    kind = _snapshot_kind(model.kind)
    return ConfigSnapshot(
        id=model.id,
        kind=kind,
        schema_version=model.schema_version,
        created_at=model.created_at,
        config=load_config_json(model.config_json),
        config_json=model.config_json,
        config_hash=model.config_hash,
    )


def _snapshot_kind(value: str) -> ConfigSnapshotKind:
    if value in ("run", "draft", "template", "import"):
        return value
    raise ValueError(f"unsupported config snapshot kind: {value}")
