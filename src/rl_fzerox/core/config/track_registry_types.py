# src/rl_fzerox/core/config/track_registry_types.py
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class RegistryRoots:
    """Config subdirectories used by the track registry."""

    courses: str = "courses"
    tracks: str = "tracks"
    vehicles: str = "vehicles"


@dataclass(frozen=True, slots=True)
class RegistryKeys:
    """YAML keys accepted by registry expansion."""

    baseline: str = "baseline"
    course_ref: str = "course_ref"
    courses: str = "courses"
    cup: str = "cup"
    entries: str = "entries"
    id: str = "id"
    weight: str = "weight"


@dataclass(frozen=True, slots=True)
class RegistrySchema:
    """Centralized registry vocabulary instead of scattered string constants."""

    roots: RegistryRoots = field(default_factory=RegistryRoots)
    keys: RegistryKeys = field(default_factory=RegistryKeys)


@dataclass(frozen=True, slots=True)
class CourseSelection:
    """One course selected from the course registry for a baseline variant."""

    ref: str
    weight: float | None = None


@dataclass(frozen=True, slots=True)
class BaselineVariant:
    """Concrete race-start baseline flavor applied to selected courses."""

    mode: str
    vehicle: str
    engine_setting: str


REGISTRY = RegistrySchema()
