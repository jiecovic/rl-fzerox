# src/rl_fzerox/core/manager/schema.py
from __future__ import annotations

import sqlite3

from rl_fzerox.core.manager.config import default_managed_run_config
from rl_fzerox.core.manager.serialization import config_hash, config_json

SCHEMA_VERSION = 1


def initialize_manager_schema(connection: sqlite3.Connection, *, applied_at: str) -> None:
    """Create manager tables and seed the built-in first template."""

    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER NOT NULL PRIMARY KEY,
            applied_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS runs (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            status TEXT NOT NULL,
            config_json TEXT NOT NULL,
            config_hash TEXT NOT NULL,
            run_dir TEXT NOT NULL UNIQUE,
            parent_run_id TEXT,
            source_run_id TEXT,
            source_artifact TEXT,
            created_at TEXT NOT NULL,
            started_at TEXT,
            stopped_at TEXT,
            FOREIGN KEY(parent_run_id) REFERENCES runs(id),
            FOREIGN KEY(source_run_id) REFERENCES runs(id)
        );

        CREATE TABLE IF NOT EXISTS run_templates (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            config_json TEXT NOT NULL,
            config_hash TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS run_drafts (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            config_json TEXT NOT NULL,
            config_hash TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS run_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            kind TEXT NOT NULL,
            message TEXT NOT NULL,
            FOREIGN KEY(run_id) REFERENCES runs(id)
        );

        CREATE UNIQUE INDEX IF NOT EXISTS runs_name_unique_idx
        ON runs(name COLLATE NOCASE);

        CREATE UNIQUE INDEX IF NOT EXISTS run_drafts_name_unique_idx
        ON run_drafts(name COLLATE NOCASE);
        """
    )
    connection.execute(
        """
        INSERT OR IGNORE INTO schema_version(version, applied_at)
        VALUES (?, ?)
        """,
        (SCHEMA_VERSION, applied_at),
    )
    _insert_default_template(connection, created_at=applied_at)


def _insert_default_template(connection: sqlite3.Connection, *, created_at: str) -> None:
    config = default_managed_run_config()
    connection.execute(
        """
        INSERT INTO run_templates(
            id,
            name,
            config_json,
            config_hash,
            created_at,
            updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            name = excluded.name,
            config_json = excluded.config_json,
            config_hash = excluded.config_hash,
            updated_at = excluded.updated_at
        """,
        (
            "all_cups_recurrent_ppo",
            "All cups recurrent PPO",
            config_json(config),
            config_hash(config),
            created_at,
            created_at,
        ),
    )
