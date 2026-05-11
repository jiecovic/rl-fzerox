# src/rl_fzerox/core/manager/storage/schema.py
"""Current-schema bootstrap for the SQLite-backed manager registry."""

from __future__ import annotations

import sqlite3

from rl_fzerox.core.manager.run_spec import default_managed_run_config
from rl_fzerox.core.manager.storage.serialization import config_hash, config_json

SCHEMA_VERSION = 12


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
            lineage_id TEXT,
            lineage_step_offset INTEGER NOT NULL DEFAULT 0,
            parent_run_id TEXT,
            source_run_id TEXT,
            source_artifact TEXT,
            source_snapshot_dir TEXT,
            source_num_timesteps INTEGER,
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
            source_run_id TEXT,
            source_artifact TEXT,
            source_snapshot_dir TEXT,
            source_num_timesteps INTEGER,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(source_run_id) REFERENCES runs(id)
        );

        CREATE TABLE IF NOT EXISTS run_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            kind TEXT NOT NULL,
            message TEXT NOT NULL,
            FOREIGN KEY(run_id) REFERENCES runs(id)
        );

        CREATE TABLE IF NOT EXISTS run_runtime (
            run_id TEXT PRIMARY KEY,
            total_timesteps INTEGER NOT NULL,
            num_timesteps INTEGER NOT NULL,
            progress_fraction REAL NOT NULL,
            updated_at TEXT NOT NULL,
            fps REAL,
            episode_reward_mean REAL,
            episode_length_mean REAL,
            approx_kl REAL,
            entropy_loss REAL,
            value_loss REAL,
            policy_gradient_loss REAL,
            FOREIGN KEY(run_id) REFERENCES runs(id)
        );

        CREATE TABLE IF NOT EXISTS run_commands (
            run_id TEXT PRIMARY KEY,
            command TEXT NOT NULL,
            requested_at TEXT NOT NULL,
            FOREIGN KEY(run_id) REFERENCES runs(id)
        );

        CREATE TABLE IF NOT EXISTS run_workers (
            run_id TEXT PRIMARY KEY,
            launch_token TEXT NOT NULL,
            pid INTEGER NOT NULL,
            launched_at TEXT NOT NULL,
            heartbeat_at TEXT NOT NULL,
            FOREIGN KEY(run_id) REFERENCES runs(id)
        );

        CREATE TABLE IF NOT EXISTS filesystem_operations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            kind TEXT NOT NULL,
            source_path TEXT NOT NULL,
            target_path TEXT,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS lineage_groups (
            lineage_id TEXT NOT NULL,
            group_name TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY(lineage_id, group_name)
        );

        DROP INDEX IF EXISTS runs_name_unique_idx;

        CREATE UNIQUE INDEX IF NOT EXISTS run_drafts_name_unique_idx
        ON run_drafts(name COLLATE NOCASE);
        """
    )
    connection.execute("DROP INDEX IF EXISTS run_metric_samples_run_id_created_at_idx")
    connection.execute("DROP TABLE IF EXISTS run_metric_samples")
    connection.execute("DELETE FROM schema_version")
    connection.execute(
        """
        INSERT INTO schema_version(version, applied_at)
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
