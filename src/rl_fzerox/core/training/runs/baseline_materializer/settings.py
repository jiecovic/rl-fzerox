# src/rl_fzerox/core/training/runs/baseline_materializer/settings.py
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from rl_fzerox.core.config.paths import project_root_dir


@dataclass(frozen=True, slots=True)
class BaselineMaterializerSettings:
    """Stable materializer settings that affect cache identity and filenames."""

    schema_version: int = 8
    boot_menu_time_attack_mode: str = "boot_menu_time_attack"
    cache_root: Path = field(
        default_factory=lambda: project_root_dir() / "local" / "cache" / "baseline_materializer"
    )
    safe_name_pattern: re.Pattern[str] = field(
        default_factory=lambda: re.compile(r"[^a-zA-Z0-9_.-]+")
    )
    cache_lock_timeout_seconds: float = 600.0
    cache_lock_poll_seconds: float = 0.1


BASELINE_MATERIALIZER_SETTINGS = BaselineMaterializerSettings()
