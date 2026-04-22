# src/rl_fzerox/core/config/track_registry.py
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from omegaconf import OmegaConf

from rl_fzerox.core.config.track_registry_types import (
    REGISTRY,
    BaselineVariant,
    CourseSelection,
)
from rl_fzerox.core.config.vehicle_catalog import (
    engine_setting_display_name,
    known_vehicle_ids,
    resolve_engine_setting,
    try_resolve_engine_setting_id,
    vehicle_by_id,
)
from rl_fzerox.core.domain.courses import (
    built_in_course_by_ref,
    built_in_course_configs,
    built_in_course_ref_by_id,
)


def expand_track_registry_metadata(
    config_data: dict[str, object],
    *,
    config_root: Path,
) -> None:
    """Expand compact course selections and enrich concrete track metadata."""

    _expand_track_course_metadata(_nested_mapping(config_data, "track"), config_root)
    _expand_track_sampling_section(
        _nested_mapping(config_data, "env", "track_sampling"),
        config_root,
    )

    curriculum = _nested_mapping(config_data, "curriculum")
    stages = curriculum.get("stages") if curriculum is not None else None
    if not isinstance(stages, list | tuple):
        return
    for stage in stages:
        if isinstance(stage, dict):
            _expand_track_sampling_section(_nested_mapping(stage, "track_sampling"), config_root)


def _expand_track_sampling_section(
    section: dict[str, object] | None,
    config_root: Path,
) -> None:
    if section is None:
        return

    raw_courses = section.pop(REGISTRY.keys.courses, None)
    raw_baseline_spec = section.pop(REGISTRY.keys.baseline, None)

    if raw_courses is not None:
        if section.get(REGISTRY.keys.entries):
            raise ValueError("track_sampling.courses cannot be combined with entries")
        section[REGISTRY.keys.entries] = _entries_from_courses(
            raw_courses,
            raw_baseline_spec,
            config_root=config_root,
        )
        return

    _expand_concrete_entries(section, config_root)


def _expand_concrete_entries(section: dict[str, object], config_root: Path) -> None:
    entries = section.get(REGISTRY.keys.entries)
    if not isinstance(entries, list | tuple):
        return
    section[REGISTRY.keys.entries] = [
        _entry_with_registry_metadata(entry, config_root=config_root) for entry in entries
    ]


def _entries_from_courses(
    raw_courses: object,
    raw_baseline_spec: object,
    *,
    config_root: Path,
) -> list[dict[str, object]]:
    if not isinstance(raw_courses, list | tuple):
        raise TypeError("track_sampling.courses must be a list")
    baseline_variant = _baseline_variant(raw_baseline_spec, config_root=config_root)
    entries: list[dict[str, object]] = []
    for raw_course in raw_courses:
        selection = _course_selection(raw_course, config_root=config_root)
        entries.append(
            _entry_from_course_variant(
                course_ref=selection.ref,
                variant=baseline_variant,
                weight=selection.weight,
                config_root=config_root,
            )
        )
    return entries


def _entry_from_course_variant(
    *,
    course_ref: str,
    variant: BaselineVariant,
    weight: float | None,
    config_root: Path,
) -> dict[str, object]:
    course = _load_course_config(course_ref, config_root=config_root)
    vehicle = vehicle_by_id(variant.vehicle)
    vehicle_name = vehicle.display_name
    engine_display_name = engine_setting_display_name(variant.engine_setting)
    course_id = _optional_str(course.get("id")) or _safe_id(course_ref)
    mode_id = _safe_id(variant.mode)

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


def _course_selection(raw_course: object, *, config_root: Path) -> CourseSelection:
    if isinstance(raw_course, str):
        return CourseSelection(ref=_course_ref_by_id(raw_course, config_root=config_root))
    if not isinstance(raw_course, Mapping):
        raise TypeError("track_sampling.courses entries must be strings or mappings")

    course_id = raw_course.get(REGISTRY.keys.id)
    if not isinstance(course_id, str) or not course_id:
        raise ValueError("course selection mappings must define a non-empty id")
    cup = raw_course.get(REGISTRY.keys.cup)
    if cup is not None and (not isinstance(cup, str) or not cup):
        raise ValueError("course selection cup must be a non-empty string when set")
    weight = _optional_weight(raw_course.get(REGISTRY.keys.weight), label="course weight")
    return CourseSelection(
        ref=_course_ref_by_id(course_id, cup=cup, config_root=config_root),
        weight=weight,
    )


def _course_ref_by_id(
    course_id: str,
    *,
    config_root: Path,
    cup: str | None = None,
) -> str:
    matches = list(built_in_course_ref_by_id(course_id, cup=cup))
    matches.extend(
        ref
        for ref, course in _iter_external_course_configs(config_root=config_root)
        if course.get("id") == course_id and (cup is None or course.get("cup") == cup)
    )
    if not matches:
        qualifier = f" in cup {cup!r}" if cup is not None else ""
        raise FileNotFoundError(f"Course registry id not found{qualifier}: {course_id!r}")
    if len(matches) > 1:
        raise ValueError(f"Course id {course_id!r} is ambiguous; use a mapping with id and cup")
    return matches[0]


def _baseline_variant(raw_baseline_spec: object, *, config_root: Path) -> BaselineVariant:
    if not isinstance(raw_baseline_spec, Mapping):
        raise TypeError("track_sampling.baseline must be a mapping")
    mode = _required_baseline_field(raw_baseline_spec, "mode")
    vehicle = _required_baseline_field(raw_baseline_spec, "vehicle")
    if "ghost" in raw_baseline_spec:
        raise ValueError(
            "track_sampling.baseline.ghost is not configurable; use no-ghost baselines"
        )
    engine_setting = resolve_engine_setting(
        raw_baseline_spec.get("engine_setting"),
        context=f"track_sampling.baseline.vehicle={vehicle!r}",
    )
    return BaselineVariant(
        mode=mode,
        vehicle=vehicle,
        engine_setting=engine_setting.id,
        engine_setting_raw_value=engine_setting.raw_value,
    )


def _required_baseline_field(raw_baseline_spec: Mapping[object, object], key: str) -> str:
    value = raw_baseline_spec.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"track_sampling.baseline.{key} must be a non-empty string")
    return value


def _entry_with_registry_metadata(
    raw_entry: object,
    *,
    config_root: Path,
) -> object:
    if not isinstance(raw_entry, dict):
        return raw_entry
    entry = {str(key): value for key, value in raw_entry.items() if isinstance(key, str)}
    enriched = _track_with_registry_metadata(entry, config_root=config_root)
    registry_entry = _registry_track_by_id(enriched.get("id"), config_root=config_root)
    if registry_entry is not None:
        registry_entry = _track_with_registry_metadata(registry_entry, config_root=config_root)
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
    elif _looks_like_legacy_generated_entry(enriched):
        legacy_metadata = _legacy_generated_entry_metadata_from_id(
            enriched,
            config_root=config_root,
        )
        if legacy_metadata is not None:
            for key, value in legacy_metadata.items():
                enriched.setdefault(key, value)
    return enriched


def _looks_like_legacy_generated_entry(entry: Mapping[str, object]) -> bool:
    """Return true for old run manifests that persisted only generated IDs."""

    return (
        entry.get("baseline_state_path") is not None
        and entry.get("course_index") is None
        and entry.get("mode") is None
        and entry.get("vehicle") is None
        and entry.get("engine_setting") is None
    )


def _legacy_generated_entry_metadata_from_id(
    entry: dict[str, object],
    *,
    config_root: Path,
) -> dict[str, object] | None:
    """Recover metadata from old generated IDs.

    Fresh configs and manifests should carry explicit course/vehicle/engine
    fields. This fallback only keeps older v4 run manifests watchable.
    """

    raw_id = entry.get("id")
    if not isinstance(raw_id, str) or not raw_id:
        return None

    for course_ref, course in _iter_course_configs(config_root=config_root):
        course_id = _optional_str(course.get("id")) or _safe_id(course_ref)
        prefix = f"{course_id}_"
        if not raw_id.startswith(prefix):
            continue
        variant = _baseline_variant_from_generated_entry_id(
            raw_id.removeprefix(prefix),
            config_root=config_root,
        )
        if variant is None:
            continue
        return _entry_from_course_variant(
            course_ref=course_ref,
            variant=variant,
            weight=_optional_weight(entry.get("weight"), label="entry weight"),
            config_root=config_root,
        )
    return None


def _baseline_variant_from_generated_entry_id(
    suffix: str,
    *,
    config_root: Path,
) -> BaselineVariant | None:
    mode = "time_attack"
    mode_prefix = f"{_safe_id(mode)}_"
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


def _expand_track_course_metadata(
    track: dict[str, object] | None,
    config_root: Path,
) -> None:
    if track is None:
        return
    enriched = _track_with_registry_metadata(track, config_root=config_root)
    track.clear()
    track.update(enriched)


def _track_with_registry_metadata(
    track: dict[str, object],
    *,
    config_root: Path,
) -> dict[str, object]:
    enriched = {str(key): value for key, value in track.items() if isinstance(key, str)}
    course_ref = enriched.get(REGISTRY.keys.course_ref)
    if isinstance(course_ref, str) and course_ref:
        course = _load_course_config(course_ref, config_root=config_root)
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


def _registry_track_by_id(raw_id: object, *, config_root: Path) -> dict[str, object] | None:
    if not isinstance(raw_id, str) or not raw_id:
        return None
    for _, track in _iter_track_configs(config_root=config_root):
        if track.get("id") == raw_id:
            return track
    return None


def _iter_track_configs(*, config_root: Path) -> tuple[tuple[str, dict[str, object]], ...]:
    registry_root = (config_root / REGISTRY.roots.tracks).resolve()
    if not registry_root.is_dir():
        return ()
    tracks: list[tuple[str, dict[str, object]]] = []
    for path in sorted(registry_root.rglob("*.yaml")):
        ref = path.relative_to(registry_root).with_suffix("").as_posix()
        loaded = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        if not isinstance(loaded, dict):
            continue
        track = loaded.get("track")
        if isinstance(track, dict):
            tracks.append((ref, {str(key): value for key, value in track.items()}))
    return tuple(tracks)


def _iter_course_configs(*, config_root: Path) -> tuple[tuple[str, dict[str, object]], ...]:
    built_in_courses = built_in_course_configs()
    external_courses = {
        ref: course for ref, course in _iter_external_course_configs(config_root=config_root)
    }
    return (*built_in_courses, *external_courses.items())


def _iter_external_course_configs(
    *,
    config_root: Path,
) -> tuple[tuple[str, dict[str, object]], ...]:
    registry_root = (config_root / REGISTRY.roots.external_courses).resolve()
    if not registry_root.is_dir():
        return ()
    courses: list[tuple[str, dict[str, object]]] = []
    for path in sorted(registry_root.rglob("*.yaml")):
        ref = path.relative_to(registry_root).with_suffix("").as_posix()
        course = _load_external_course_config(ref, config_root=config_root)
        if course is not None:
            courses.append((ref, course))
    return tuple(courses)


def _load_course_config(ref: str, *, config_root: Path) -> dict[str, object]:
    built_in = built_in_course_by_ref(ref)
    if built_in is not None:
        return built_in
    external = _load_external_course_config(ref, config_root=config_root)
    if external is not None:
        return external
    raise FileNotFoundError(f"Course registry entry not found: {ref!r}")


def _load_external_course_config(ref: str, *, config_root: Path) -> dict[str, object] | None:
    path = _registry_path(
        root=config_root / REGISTRY.roots.external_courses,
        ref=ref,
        label="Course registry entry",
        required=False,
    )
    if path is None:
        return None
    loaded = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(loaded, dict):
        raise TypeError(f"Course registry entry {ref!r} must resolve to a mapping")
    course = loaded.get("course")
    if not isinstance(course, dict):
        raise ValueError(f"Course registry entry {ref!r} does not define a course section")
    return {str(key): value for key, value in course.items() if isinstance(key, str)}


def _safe_id(value: str) -> str:
    return value.replace("/", "_").replace("-", "_")


def _optional_weight(raw_weight: object, *, label: str) -> float | None:
    if raw_weight is None:
        return None
    if isinstance(raw_weight, bool) or not isinstance(raw_weight, int | float):
        raise TypeError(f"{label} must be numeric")
    weight = float(raw_weight)
    if weight <= 0.0:
        raise ValueError(f"{label} must be greater than zero")
    return weight


def _registry_path(*, root: Path, ref: str, label: str, required: bool = True) -> Path | None:
    registry_root = root.resolve()
    path = (registry_root / ref).with_suffix(".yaml").resolve()
    if not path.is_relative_to(registry_root):
        raise ValueError(f"{label} ref escapes registry root: {ref!r}")
    if not path.is_file():
        if not required:
            return None
        raise FileNotFoundError(f"{label} not found: {ref!r}")
    return path


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _nested_mapping(value: dict[str, object], *path: str) -> dict[str, object] | None:
    cursor: object = value
    for key in path:
        if not isinstance(cursor, dict):
            return None
        cursor = cursor.get(key)
    return cursor if isinstance(cursor, dict) else None
