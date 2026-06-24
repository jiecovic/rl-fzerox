# src/rl_fzerox/core/manager/db/session.py
"""Session lifecycle for manager DB repository operations."""

from __future__ import annotations

import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session


def manager_engine(db_path: Path) -> Engine:
    """Create one SQLite engine with the manager DB pragmas."""

    engine = create_engine(
        f"sqlite:///{db_path}",
        future=True,
        connect_args={"timeout": 30.0, "check_same_thread": False},
    )

    @event.listens_for(engine, "connect")
    def _configure_connection(
        dbapi_connection: sqlite3.Connection,
        _connection_record: object,
    ) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode = WAL")
        cursor.execute("PRAGMA busy_timeout = 30000")
        cursor.execute("PRAGMA foreign_keys = ON")
        cursor.close()

    return engine


@contextmanager
def manager_session(db_path: Path) -> Generator[Session]:
    """Open one transactional SQLAlchemy session for a manager operation."""

    engine = manager_engine(db_path)
    try:
        with Session(engine, expire_on_commit=False) as session:
            with session.begin():
                yield session
    finally:
        engine.dispose()
