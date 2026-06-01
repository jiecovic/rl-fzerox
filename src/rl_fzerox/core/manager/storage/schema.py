# src/rl_fzerox/core/manager/storage/schema.py
"""Current-schema bootstrap for the SQLite-backed manager registry."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import inspect, select
from sqlalchemy.engine import Engine
from sqlalchemy.engine.reflection import Inspector
from sqlalchemy.orm import Session

from rl_fzerox.core.manager.db import manager_session
from rl_fzerox.core.manager.db.models import ManagerBase, SchemaVersionModel
from rl_fzerox.core.manager.db.repositories.configs import create_config_snapshot
from rl_fzerox.core.manager.db.repositories.runs import upsert_template
from rl_fzerox.core.manager.db.session import manager_engine
from rl_fzerox.core.manager.run_spec import default_managed_run_config
from rl_fzerox.core.manager.storage.serialization import config_hash

SCHEMA_VERSION = 17

CONFIG_OWNER_TABLES = ("runs", "run_drafts", "run_templates")
RUN_CHILD_TABLES = (
    "run_commands",
    "run_events",
    "run_runtime",
    "run_track_sampling_entries",
    "run_track_sampling_runtime",
    "run_workers",
)


def initialize_manager_schema(db_path: Path, *, applied_at: str) -> None:
    """Create manager tables and seed the built-in first template."""

    db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = manager_engine(db_path)
    try:
        table_names = set(inspect(engine).get_table_names())
        if _has_manager_schema(table_names):
            _assert_current_schema(engine=engine, table_names=table_names)
        else:
            ManagerBase.metadata.create_all(engine)
    finally:
        engine.dispose()

    with manager_session(db_path) as session:
        for version in session.scalars(select(SchemaVersionModel)):
            session.delete(version)
        session.add(SchemaVersionModel(version=SCHEMA_VERSION, applied_at=applied_at))
        _refresh_default_template(session=session, updated_at=applied_at)


def refresh_default_template(db_path: Path, *, updated_at: str) -> None:
    """Refresh the built-in template without re-running schema creation."""

    with manager_session(db_path) as session:
        _refresh_default_template(session=session, updated_at=updated_at)


def _refresh_default_template(
    *,
    session: Session,
    updated_at: str,
) -> None:
    config = default_managed_run_config()
    snapshot = create_config_snapshot(
        session,
        kind="template",
        config=config,
        created_at=updated_at,
        snapshot_id=f"cfg_template_all_cups_recurrent_ppo_{config_hash(config)[:12]}",
    )
    upsert_template(
        session,
        template_id="all_cups_recurrent_ppo",
        name="All cups recurrent PPO",
        config_snapshot_id=snapshot.id,
        created_at=updated_at,
        updated_at=updated_at,
    )


def _has_manager_schema(table_names: set[str]) -> bool:
    return bool(
        table_names
        & {
            "schema_version",
            "runs",
            "run_drafts",
            "run_templates",
            "run_runtime",
        }
    )


def _assert_current_schema(
    *,
    engine: Engine,
    table_names: set[str],
) -> None:
    inspector = inspect(engine)
    for table_name in CONFIG_OWNER_TABLES:
        if table_name not in table_names:
            raise RuntimeError(f"manager DB is not current: missing {table_name}")
        columns = {column["name"] for column in inspector.get_columns(table_name)}
        if "config_json" in columns or "config_hash" in columns:
            raise RuntimeError(
                "manager DB is not current: config JSON must be stored in config_snapshots only"
            )
        if "config_snapshot_id" not in columns:
            raise RuntimeError(
                f"manager DB is not current: {table_name} is missing config_snapshot_id"
            )
    if "config_snapshots" not in table_names:
        raise RuntimeError("manager DB is not current: missing config_snapshots")
    _assert_run_foreign_keys(inspector=inspector, table_names=table_names)


def _assert_run_foreign_keys(
    *,
    inspector: Inspector,
    table_names: set[str],
) -> None:
    for table_name in RUN_CHILD_TABLES:
        if table_name not in table_names:
            raise RuntimeError(f"manager DB is not current: missing {table_name}")
        for foreign_key in inspector.get_foreign_keys(table_name):
            if "run_id" not in foreign_key.get("constrained_columns", ()):
                continue
            if foreign_key.get("referred_table") != "runs":
                raise RuntimeError(
                    f"manager DB is not current: {table_name}.run_id must reference runs.id"
                )
