# src/rl_fzerox/core/runtime_spec/track_registry/selection.py
from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

from rl_fzerox.core.domain.courses import built_in_course_ref_by_id, built_in_course_refs_by_cup
from rl_fzerox.core.domain.race_difficulty import (
    RaceDifficultyName,
    default_gp_difficulty,
    is_race_difficulty_name,
    race_difficulty_names,
)
from rl_fzerox.core.runtime_spec.track_registry_types import (
    REGISTRY,
    BaselineVariant,
    CourseSelection,
)
from rl_fzerox.core.runtime_spec.vehicle_catalog import resolve_engine_setting

from .common import optional_weight
from .registry import iter_external_course_configs


def entries_from_courses(
    raw_courses: object,
    raw_baseline_spec: object,
    *,
    config_root: Path,
    entry_from_course_variant: Callable[..., dict[str, object]],
) -> list[dict[str, object]]:
    if not isinstance(raw_courses, list | tuple):
        raise TypeError("track_sampling.courses must be a list")
    variant = baseline_variant(raw_baseline_spec, config_root=config_root)
    entries: list[dict[str, object]] = []
    for raw_course in raw_courses:
        for selection in course_selections(raw_course, config_root=config_root):
            entries.append(
                entry_from_course_variant(
                    course_ref=selection.ref,
                    variant=variant,
                    weight=selection.weight,
                    config_root=config_root,
                )
            )
    return entries


def course_selections(raw_course: object, *, config_root: Path) -> tuple[CourseSelection, ...]:
    """Resolve one course selector, including cup-wide selectors."""

    if isinstance(raw_course, str):
        return (CourseSelection(ref=course_ref_by_id(raw_course, config_root=config_root)),)
    if not isinstance(raw_course, Mapping):
        raise TypeError("track_sampling.courses entries must be strings or mappings")

    course_id = raw_course.get(REGISTRY.keys.id)
    cup = raw_course.get(REGISTRY.keys.cup)
    if cup is not None and (not isinstance(cup, str) or not cup):
        raise ValueError("course selection cup must be a non-empty string when set")
    weight = optional_weight(raw_course.get(REGISTRY.keys.weight), label="course weight")

    if isinstance(course_id, str) and course_id:
        return (
            CourseSelection(
                ref=course_ref_by_id(course_id, cup=cup, config_root=config_root),
                weight=weight,
            ),
        )
    if course_id is not None:
        raise ValueError("course selection id must be a non-empty string when set")
    if isinstance(cup, str):
        return tuple(
            CourseSelection(ref=ref, weight=weight)
            for ref in course_refs_by_cup(cup, config_root=config_root)
        )
    raise ValueError("course selection mappings must define a non-empty id or cup")


def course_ref_by_id(
    course_id: str,
    *,
    config_root: Path,
    cup: str | None = None,
) -> str:
    matches = list(built_in_course_ref_by_id(course_id, cup=cup))
    matches.extend(
        ref
        for ref, course in iter_external_course_configs(config_root=config_root)
        if course.get("id") == course_id and (cup is None or course.get("cup") == cup)
    )
    if not matches:
        qualifier = f" in cup {cup!r}" if cup is not None else ""
        raise FileNotFoundError(f"Course registry id not found{qualifier}: {course_id!r}")
    if len(matches) > 1:
        raise ValueError(f"Course id {course_id!r} is ambiguous; use a mapping with id and cup")
    return matches[0]


def course_refs_by_cup(cup: str, *, config_root: Path) -> tuple[str, ...]:
    matches = list(built_in_course_refs_by_cup(cup))
    matches.extend(
        ref
        for ref, course in iter_external_course_configs(config_root=config_root)
        if course.get("cup") == cup
    )
    if not matches:
        raise FileNotFoundError(f"Course registry cup not found: {cup!r}")
    return tuple(matches)


def baseline_variant(raw_baseline_spec: object, *, config_root: Path) -> BaselineVariant:
    del config_root
    if not isinstance(raw_baseline_spec, Mapping):
        raise TypeError("track_sampling.baseline must be a mapping")
    mode = required_baseline_field(raw_baseline_spec, "mode")
    vehicle = required_baseline_field(raw_baseline_spec, "vehicle")
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
        gp_difficulty=_baseline_gp_difficulty(raw_baseline_spec, mode=mode),
        vehicle=vehicle,
        engine_setting=engine_setting.id,
        engine_setting_raw_value=engine_setting.raw_value,
    )


def required_baseline_field(raw_baseline_spec: Mapping[object, object], key: str) -> str:
    value = raw_baseline_spec.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"track_sampling.baseline.{key} must be a non-empty string")
    return value


def _baseline_gp_difficulty(
    raw_baseline_spec: Mapping[object, object],
    *,
    mode: str,
) -> RaceDifficultyName | None:
    if mode != "gp_race":
        return None
    value = raw_baseline_spec.get("gp_difficulty")
    if value is None:
        return default_gp_difficulty()
    if not isinstance(value, str) or not value:
        raise ValueError("track_sampling.baseline.gp_difficulty must be a non-empty string")
    if not is_race_difficulty_name(value):
        allowed = ", ".join(race_difficulty_names())
        raise ValueError(f"track_sampling.baseline.gp_difficulty must be one of {allowed}")
    return value
