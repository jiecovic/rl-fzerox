# src/rl_fzerox/core/config/track_registry/metadata.py
from __future__ import annotations

from pathlib import Path

from rl_fzerox.core.config.track_registry_types import BaselineVariant
from rl_fzerox.core.config.vehicle_catalog import (
    engine_setting_display_name,
    resolve_engine_setting,
    vehicle_by_id,
)

from .common import optional_str, optional_weight, safe_id
from .legacy import baseline_variant_from_generated_entry_id, looks_like_legacy_generated_entry
from .registry import iter_course_configs, load_course_config, registry_track_by_id


def enrich_concrete_entries(section: dict[str, object], config_root: Path) -> None:
    entries = section.get("entries")
    if not isinstance(entries, list | tuple):
        return
    section["entries"] = [
        enrich_entry_with_registry_metadata(entry, config_root=config_root) for entry in entries
    ]


def entry_from_course_variant(
    *,
    course_ref: str,
    variant: BaselineVariant,
    weight: float | None,
    config_root: Path,
) -> dict[str, object]:
    course = load_course_config(course_ref, config_root=config_root)
    vehicle = vehicle_by_id(variant.vehicle)
    vehicle_name = vehicle.display_name
    engine_display_name = engine_setting_display_name(variant.engine_setting)
    course_id = optional_str(course.get("id")) or safe_id(course_ref)
    mode_id = safe_id(variant.mode)

    entry: dict[str, object] = {
        "id": f"{course_id}_{mode_id}_{variant.vehicle}_{variant.engine_setting}",
        "display_name": (
            f"{course.get('display_name', course_id)} "
            f"{variant.mode.replace('_', ' ').title()} - "
            f"{vehicle_name} {engine_display_name}"
        ),
        "course_ref": course_ref,
        "course_id": course_id,
        "course_name": course.get("display_name"),
        "course_index": course.get("course_index"),
        "mode": variant.mode,
        "vehicle": variant.vehicle,
        "vehicle_name": vehicle_name,
        "engine_setting": variant.engine_setting,
        "engine_setting_raw_value": variant.engine_setting_raw_value,
    }
    if "records" in course:
        entry["records"] = course["records"]
    if weight is not None:
        entry["weight"] = weight
    return entry


def enrich_entry_with_registry_metadata(
    raw_entry: object,
    *,
    config_root: Path,
) -> object:
    if not isinstance(raw_entry, dict):
        return raw_entry
    entry = {str(key): value for key, value in raw_entry.items() if isinstance(key, str)}
    enriched = enrich_track_with_registry_metadata(entry, config_root=config_root)
    registry_entry = registry_track_by_id(enriched.get("id"), config_root=config_root)
    if registry_entry is not None:
        registry_entry = enrich_track_with_registry_metadata(
            registry_entry,
            config_root=config_root,
        )
        for key, value in registry_entry.items():
            if key in {
                "display_name",
                "course_ref",
                "course_id",
                "course_name",
                "course_index",
                "mode",
                "vehicle",
                "vehicle_name",
                "source_vehicle",
                "engine_setting",
                "engine_setting_raw_value",
                "source_course_index",
                "source_engine_setting",
                "source_engine_setting_raw_value",
                "records",
            }:
                enriched.setdefault(key, value)
    elif looks_like_legacy_generated_entry(enriched):
        legacy_metadata = legacy_generated_entry_metadata_from_id(
            enriched,
            config_root=config_root,
        )
        if legacy_metadata is not None:
            for key, value in legacy_metadata.items():
                enriched.setdefault(key, value)
    return enriched


def legacy_generated_entry_metadata_from_id(
    entry: dict[str, object],
    *,
    config_root: Path,
) -> dict[str, object] | None:
    """Recover metadata from old generated IDs."""

    raw_id = entry.get("id")
    if not isinstance(raw_id, str) or not raw_id:
        return None

    for course_ref, course in iter_course_configs(config_root=config_root):
        course_id = optional_str(course.get("id")) or safe_id(course_ref)
        prefix = f"{course_id}_"
        if not raw_id.startswith(prefix):
            continue
        variant = baseline_variant_from_generated_entry_id(
            raw_id.removeprefix(prefix),
            config_root=config_root,
        )
        if variant is None:
            continue
        return entry_from_course_variant(
            course_ref=course_ref,
            variant=variant,
            weight=optional_weight(entry.get("weight"), label="entry weight"),
            config_root=config_root,
        )
    return None


def enrich_track_with_registry_metadata(
    track: dict[str, object],
    *,
    config_root: Path,
) -> dict[str, object]:
    enriched = {str(key): value for key, value in track.items() if isinstance(key, str)}
    course_ref = enriched.get("course_ref")
    if isinstance(course_ref, str) and course_ref:
        course = load_course_config(course_ref, config_root=config_root)
        for source_key, target_key in (
            ("id", "course_id"),
            ("display_name", "course_name"),
            ("course_index", "course_index"),
            ("records", "records"),
        ):
            if source_key in course:
                enriched.setdefault(target_key, course[source_key])
        if "course_index" in enriched:
            enriched.setdefault("source_course_index", enriched["course_index"])

    vehicle = enriched.get("vehicle")
    if isinstance(vehicle, str) and vehicle:
        vehicle_info = vehicle_by_id(vehicle)
        engine_setting = enriched.get("engine_setting")
        if (isinstance(engine_setting, str) and engine_setting) or (
            isinstance(engine_setting, int) and not isinstance(engine_setting, bool)
        ):
            resolved_engine_setting = resolve_engine_setting(
                engine_setting,
                context=f"track vehicle={vehicle!r}",
            )
            enriched["engine_setting"] = resolved_engine_setting.id
            if resolved_engine_setting.raw_value is not None:
                enriched.setdefault("engine_setting_raw_value", resolved_engine_setting.raw_value)
                enriched.setdefault("source_engine_setting", resolved_engine_setting.id)
                enriched.setdefault(
                    "source_engine_setting_raw_value",
                    resolved_engine_setting.raw_value,
                )
        enriched.setdefault("vehicle_name", vehicle_info.display_name)
        enriched.setdefault("source_vehicle", vehicle)
    return enriched
