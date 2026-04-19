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
        track_ref = _track_ref_for_course_variant(
            course_ref=selection.ref,
            variant=baseline_variant,
            config_root=config_root,
        )
        entries.append(
            _entry_from_track_ref(
                track_ref,
                weight=selection.weight,
                config_root=config_root,
            )
        )
    return entries


def _track_ref_for_course_variant(
    *,
    course_ref: str,
    variant: BaselineVariant,
    config_root: Path,
) -> str:
    matches = [
        ref
        for ref, track in _iter_track_configs(config_root=config_root)
        if track.get("course_ref") == course_ref
        and track.get("mode") == variant.mode
        and track.get("vehicle") == variant.vehicle
        and track.get("engine_setting") == variant.engine_setting
    ]
    if not matches:
        raise FileNotFoundError(
            "Track registry entry not found for "
            f"course={course_ref!r}, mode={variant.mode!r}, "
            f"vehicle={variant.vehicle!r}, engine_setting={variant.engine_setting!r}"
        )
    if len(matches) > 1:
        raise ValueError(
            "Track registry selection is ambiguous for "
            f"course={course_ref!r}, mode={variant.mode!r}, "
            f"vehicle={variant.vehicle!r}, engine_setting={variant.engine_setting!r}"
        )
    return matches[0]


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
    matches = [
        ref
        for ref, course in _iter_course_configs(config_root=config_root)
        if course.get("id") == course_id and (cup is None or course.get("cup") == cup)
    ]
    if not matches:
        qualifier = f" in cup {cup!r}" if cup is not None else ""
        raise FileNotFoundError(f"Course registry id not found{qualifier}: {course_id!r}")
    if len(matches) > 1:
        raise ValueError(
            f"Course id {course_id!r} is ambiguous; use a mapping with id and cup"
        )
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
    vehicle_config = _vehicle_config_by_id(vehicle, config_root=config_root)
    engine_setting = _engine_setting_id(
        vehicle_config,
        raw_baseline_spec.get("engine_setting"),
        context=f"track_sampling.baseline.vehicle={vehicle!r}",
    )
    return BaselineVariant(
        mode=mode,
        vehicle=vehicle,
        engine_setting=engine_setting,
    )


def _required_baseline_field(raw_baseline_spec: Mapping[object, object], key: str) -> str:
    value = raw_baseline_spec.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"track_sampling.baseline.{key} must be a non-empty string")
    return value


def _entry_from_track_ref(
    ref: str,
    *,
    weight: float | None = None,
    config_root: Path,
) -> dict[str, object]:
    track_config = _load_track_config(ref, config_root=config_root)
    track = track_config.get("track")
    if not isinstance(track, dict):
        raise ValueError(f"Track registry entry {ref!r} does not define a track section")
    track = _track_with_registry_metadata(track, config_root=config_root)

    entry = {
        key: track[key]
        for key in (
            "id",
            "display_name",
            "course_ref",
            "course_id",
            "course_name",
            "baseline_state_path",
            "course_index",
            "mode",
            "vehicle",
            "vehicle_name",
            "engine_setting",
            "records",
        )
        if key in track
    }
    if "id" not in entry or "baseline_state_path" not in entry:
        raise ValueError(f"Track registry entry {ref!r} must define id and baseline_state_path")
    if weight is not None:
        entry["weight"] = weight
    return entry


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
                "engine_setting",
                "records",
            }:
                enriched.setdefault(key, value)
    return enriched


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

    vehicle = enriched.get("vehicle")
    if isinstance(vehicle, str) and vehicle:
        vehicle_config = _vehicle_config_by_id(vehicle, config_root=config_root)
        engine_setting = enriched.get("engine_setting")
        if isinstance(engine_setting, str) and engine_setting:
            _engine_setting_id(
                vehicle_config,
                engine_setting,
                context=f"track vehicle={vehicle!r}",
            )
        vehicle_name = _optional_str(vehicle_config.get("display_name"))
        if vehicle_name is not None:
            enriched.setdefault("vehicle_name", vehicle_name)
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
    registry_root = (config_root / REGISTRY.roots.courses).resolve()
    if not registry_root.is_dir():
        return ()
    courses: list[tuple[str, dict[str, object]]] = []
    for path in sorted(registry_root.rglob("*.yaml")):
        ref = path.relative_to(registry_root).with_suffix("").as_posix()
        courses.append((ref, _load_course_config(ref, config_root=config_root)))
    return tuple(courses)


def _load_course_config(ref: str, *, config_root: Path) -> dict[str, object]:
    path = _registry_path(
        root=config_root / REGISTRY.roots.courses,
        ref=ref,
        label="Course registry entry",
    )
    loaded = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(loaded, dict):
        raise TypeError(f"Course registry entry {ref!r} must resolve to a mapping")
    course = loaded.get("course")
    if not isinstance(course, dict):
        raise ValueError(f"Course registry entry {ref!r} does not define a course section")
    return {str(key): value for key, value in course.items() if isinstance(key, str)}


def _vehicle_config_by_id(vehicle_id: str, *, config_root: Path) -> dict[str, object]:
    registry_root = (config_root / REGISTRY.roots.vehicles).resolve()
    direct_path = (registry_root / vehicle_id).with_suffix(".yaml").resolve()
    if direct_path.is_relative_to(registry_root) and direct_path.is_file():
        return _load_vehicle_config(direct_path, ref=vehicle_id)

    matches: list[dict[str, object]] = []
    if registry_root.is_dir():
        for path in sorted(registry_root.rglob("*.yaml")):
            vehicle = _load_vehicle_config(path, ref=path.stem)
            if vehicle.get("id") == vehicle_id:
                matches.append(vehicle)
    if not matches:
        raise FileNotFoundError(f"Vehicle registry id not found: {vehicle_id!r}")
    if len(matches) > 1:
        raise ValueError(f"Vehicle id {vehicle_id!r} is ambiguous")
    return matches[0]


def _load_vehicle_config(path: Path, *, ref: str) -> dict[str, object]:
    loaded = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(loaded, dict):
        raise TypeError(f"Vehicle registry entry {ref!r} must resolve to a mapping")
    vehicle = loaded.get("vehicle")
    if not isinstance(vehicle, dict):
        raise ValueError(f"Vehicle registry entry {ref!r} does not define a vehicle section")
    return {str(key): value for key, value in vehicle.items() if isinstance(key, str)}


def _engine_setting_id(
    vehicle_config: dict[str, object],
    raw_engine_setting: object,
    *,
    context: str,
) -> str:
    engine_settings = vehicle_config.get("engine_settings")
    if isinstance(raw_engine_setting, str) and raw_engine_setting:
        if engine_settings is None:
            return raw_engine_setting
        if not isinstance(engine_settings, Mapping):
            raise TypeError(f"{context} vehicle engine_settings must be a mapping")
        if raw_engine_setting in engine_settings:
            return raw_engine_setting
        known = ", ".join(sorted(str(key) for key in engine_settings))
        raise ValueError(
            f"{context} unknown engine_setting {raw_engine_setting!r}; known: {known}"
        )

    if isinstance(raw_engine_setting, bool) or not isinstance(raw_engine_setting, int):
        raise ValueError(
            "track_sampling.baseline.engine_setting must be a string id or raw integer"
        )
    if engine_settings is None:
        raise ValueError(f"{context} cannot resolve raw engine_setting without engine_settings")
    if not isinstance(engine_settings, Mapping):
        raise TypeError(f"{context} vehicle engine_settings must be a mapping")
    for setting_id, setting_data in engine_settings.items():
        if (
            isinstance(setting_id, str)
            and _engine_setting_raw_value(setting_data) == raw_engine_setting
        ):
            return setting_id
    known = ", ".join(
        f"{setting_id}={_engine_setting_raw_value(setting_data)}"
        for setting_id, setting_data in engine_settings.items()
        if _engine_setting_raw_value(setting_data) is not None
    )
    raise ValueError(
        f"{context} unknown raw engine_setting {raw_engine_setting!r}; known: {known}"
    )


def _engine_setting_raw_value(setting_data: object) -> int | None:
    if not isinstance(setting_data, Mapping):
        return None
    raw_value = setting_data.get("raw_value")
    if isinstance(raw_value, bool):
        return None
    return raw_value if isinstance(raw_value, int) else None


def _optional_weight(raw_weight: object, *, label: str) -> float | None:
    if raw_weight is None:
        return None
    if isinstance(raw_weight, bool) or not isinstance(raw_weight, int | float):
        raise TypeError(f"{label} must be numeric")
    weight = float(raw_weight)
    if weight <= 0.0:
        raise ValueError(f"{label} must be greater than zero")
    return weight


def _load_track_config(ref: str, *, config_root: Path) -> dict[str, object]:
    path = _registry_path(
        root=config_root / REGISTRY.roots.tracks,
        ref=ref,
        label="Track registry entry",
    )
    loaded = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(loaded, dict):
        raise TypeError(f"Track registry entry {ref!r} must resolve to a mapping")
    return {str(key): value for key, value in loaded.items() if isinstance(key, str)}


def _registry_path(*, root: Path, ref: str, label: str) -> Path:
    registry_root = root.resolve()
    path = (registry_root / ref).with_suffix(".yaml").resolve()
    if not path.is_relative_to(registry_root):
        raise ValueError(f"{label} ref escapes registry root: {ref!r}")
    if not path.is_file():
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
