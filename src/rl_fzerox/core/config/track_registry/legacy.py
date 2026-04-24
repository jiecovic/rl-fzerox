# src/rl_fzerox/core/config/track_registry/legacy.py
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from rl_fzerox.core.config.track_registry_types import BaselineVariant
from rl_fzerox.core.config.vehicle_catalog import known_vehicle_ids, try_resolve_engine_setting_id

from .common import safe_id


def looks_like_legacy_generated_entry(entry: Mapping[str, object]) -> bool:
    """Return true for old run manifests that persisted only generated IDs."""

    return (
        entry.get("baseline_state_path") is not None
        and entry.get("course_index") is None
        and entry.get("mode") is None
        and entry.get("vehicle") is None
        and entry.get("engine_setting") is None
    )


def baseline_variant_from_generated_entry_id(
    suffix: str,
    *,
    config_root: Path,
) -> BaselineVariant | None:
    del config_root
    mode = "time_attack"
    mode_prefix = f"{safe_id(mode)}_"
    if not suffix.startswith(mode_prefix):
        return None
    vehicle_and_engine = suffix.removeprefix(mode_prefix)
    for vehicle_id in sorted(known_vehicle_ids(), key=len, reverse=True):
        vehicle_prefix = f"{vehicle_id}_"
        if not vehicle_and_engine.startswith(vehicle_prefix):
            continue
        engine_setting = vehicle_and_engine.removeprefix(vehicle_prefix)
        resolved_engine_setting = try_resolve_engine_setting_id(
            engine_setting,
            context=f"track_sampling.entries.id={suffix!r}",
        )
        if resolved_engine_setting is None:
            continue
        return BaselineVariant(
            mode=mode,
            vehicle=vehicle_id,
            engine_setting=resolved_engine_setting.id,
            engine_setting_raw_value=resolved_engine_setting.raw_value,
        )
    return None
