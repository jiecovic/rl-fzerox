# src/rl_fzerox/core/manager/registry/common.py
from __future__ import annotations

import os
import re
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4

from rl_fzerox.core.manager.errors import ManagerNameConflictError
from rl_fzerox.core.manager.models import (
    RunCommand,
    RunStatus,
    SaveAttemptStatus,
    SaveGameStatus,
)


def new_run_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return f"{timestamp}-{uuid4().hex[:8]}"


def new_record_id(name: str) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    slug = slugify(name) or "run"
    return f"{timestamp}-{slug}-{uuid4().hex[:8]}"


def raise_name_conflict(
    error: sqlite3.IntegrityError,
    *,
    table: str,
    kind: str,
    name: str,
) -> None:
    if f"UNIQUE constraint failed: {table}.name" in str(error):
        raise ManagerNameConflictError(kind=kind, name=name) from error


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.strip().lower())
    return slug.strip("-")[:48]


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def pid_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def run_status(value: object) -> RunStatus:
    match value:
        case "created":
            return "created"
        case "running":
            return "running"
        case "paused":
            return "paused"
        case "stopped":
            return "stopped"
        case "finished":
            return "finished"
        case "failed":
            return "failed"
        case "archived":
            return "archived"
    raise ValueError(f"Unsupported managed run status: {value!r}")


def save_game_status(value: object) -> SaveGameStatus:
    match value:
        case "created":
            return "created"
        case "running":
            return "running"
        case "paused":
            return "paused"
        case "finished":
            return "finished"
        case "failed":
            return "failed"
    raise ValueError(f"Unsupported managed save-game status: {value!r}")


def save_attempt_status(value: object) -> SaveAttemptStatus:
    match value:
        case "running":
            return "running"
        case "succeeded":
            return "succeeded"
        case "failed":
            return "failed"
    raise ValueError(f"Unsupported managed save attempt status: {value!r}")


def run_command(value: object) -> RunCommand | None:
    match value:
        case None:
            return None
        case "pause":
            return "pause"
        case "stop":
            return "stop"
    raise ValueError(f"Unsupported managed run command: {value!r}")


def optional_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def optional_float(value: object) -> float | None:
    return float(value) if isinstance(value, int | float) else None


def optional_int(value: object) -> int | None:
    return value if isinstance(value, int) else None


def optional_source_artifact(value: object) -> Literal["latest", "best"] | None:
    if value is None:
        return None
    artifact = str(value)
    if artifact == "latest":
        return "latest"
    if artifact == "best":
        return "best"
    raise ValueError(f"Unsupported managed source artifact: {artifact!r}")


def optional_path(value: object) -> Path | None:
    return Path(str(value)).expanduser().resolve() if isinstance(value, str) else None
