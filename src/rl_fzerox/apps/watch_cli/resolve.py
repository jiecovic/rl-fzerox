# src/rl_fzerox/apps/watch_cli/resolve.py
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from rl_fzerox.apps._cli import normalize_cli_overrides
from rl_fzerox.core.manager.projection.watch import (
    default_watch_config_from_train_run,
)
from rl_fzerox.core.manager.projection.watch import (
    resolve_watch_app_config as resolve_watch_app_config_from_core,
)
from rl_fzerox.core.runtime_spec.schema import WatchAppConfig


def resolve_watch_app_config(
    *,
    run_id: str,
    policy_artifact: Literal["latest", "best"] | None,
    manager_db_path: Path | None,
    session_name: str | None = None,
    overrides: Sequence[str],
) -> WatchAppConfig:
    """Resolve watch config from CLI arguments."""

    return resolve_watch_app_config_from_core(
        run_id=run_id,
        policy_artifact=policy_artifact,
        manager_db_path=manager_db_path,
        session_name=session_name,
        overrides=normalize_cli_overrides(overrides),
    )


__all__ = [
    "default_watch_config_from_train_run",
    "resolve_watch_app_config",
]
