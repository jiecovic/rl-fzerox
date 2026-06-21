# src/rl_fzerox/ui/watch/runtime/career_mode/recording/summary/values.py
from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

from rl_fzerox.core.runtime_spec.vehicle_catalog import engine_setting_display_name_for_raw


def utc_timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _summary_json_value(value: object) -> object:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _summary_text(value: object) -> str:
    if value is None:
        return "-"
    return str(value).replace("|", "\\|")


def _format_summary_time(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, int):
        return "-"
    minutes, remainder = divmod(max(0, value), 60_000)
    seconds, millis = divmod(remainder, 1_000)
    return f"{minutes}:{seconds:02d}.{millis:03d}"


def _format_summary_engine(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, int):
        return "-"
    try:
        return engine_setting_display_name_for_raw(value)
    except ValueError:
        return _summary_text(value)


def _str_info(info: Mapping[str, object], key: str) -> str | None:
    value = info.get(key)
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _int_mapping(info: Mapping[str, object], key: str) -> int | None:
    value = info.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value
