# src/rl_fzerox/core/config/track_registry/registry.py
from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from rl_fzerox.core.config.track_registry_types import REGISTRY
from rl_fzerox.core.domain.courses import built_in_course_by_ref, built_in_course_configs

from .common import registry_path


def registry_track_by_id(raw_id: object, *, config_root: Path) -> dict[str, object] | None:
    if not isinstance(raw_id, str) or not raw_id:
        return None
    for _, track in iter_track_configs(config_root=config_root):
        if track.get("id") == raw_id:
            return track
    return None


def iter_track_configs(*, config_root: Path) -> tuple[tuple[str, dict[str, object]], ...]:
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


def iter_course_configs(*, config_root: Path) -> tuple[tuple[str, dict[str, object]], ...]:
    built_in_courses = built_in_course_configs()
    external_courses = {
        ref: course for ref, course in iter_external_course_configs(config_root=config_root)
    }
    return (*built_in_courses, *external_courses.items())


def iter_external_course_configs(
    *,
    config_root: Path,
) -> tuple[tuple[str, dict[str, object]], ...]:
    registry_root = (config_root / REGISTRY.roots.external_courses).resolve()
    if not registry_root.is_dir():
        return ()
    courses: list[tuple[str, dict[str, object]]] = []
    for path in sorted(registry_root.rglob("*.yaml")):
        ref = path.relative_to(registry_root).with_suffix("").as_posix()
        course = load_external_course_config(ref, config_root=config_root)
        if course is not None:
            courses.append((ref, course))
    return tuple(courses)


def load_course_config(ref: str, *, config_root: Path) -> dict[str, object]:
    built_in = built_in_course_by_ref(ref)
    if built_in is not None:
        return built_in
    external = load_external_course_config(ref, config_root=config_root)
    if external is not None:
        return external
    raise FileNotFoundError(f"Course registry entry not found: {ref!r}")


def load_external_course_config(ref: str, *, config_root: Path) -> dict[str, object] | None:
    path = registry_path(
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
