# src/rl_fzerox/core/runtime_spec/track_registry/expand.py
"""Entry points for expanding compact track registry config sections.

Loaders call this module before Pydantic validation so registry-backed track
metadata and course selections become explicit runtime-schema entries.
"""

from __future__ import annotations

from pathlib import Path

from rl_fzerox.core.runtime_spec.track_registry_types import REGISTRY

from .common import nested_mapping
from .metadata import (
    enrich_concrete_entries,
    enrich_track_with_registry_metadata,
    entry_from_course_variant,
)
from .selection import entries_from_courses


def expand_track_registry_metadata(
    config_data: dict[str, object],
    *,
    config_root: Path,
) -> None:
    """Expand compact course selections and enrich concrete track metadata."""

    expand_track_course_metadata(nested_mapping(config_data, "track"), config_root)
    env_track_sampling = nested_mapping(config_data, "env", "track_sampling")
    expand_track_sampling_section(
        env_track_sampling,
        config_root,
    )


def expand_track_sampling_section(
    section: dict[str, object] | None,
    config_root: Path,
    *,
    default_baseline_spec: object = None,
) -> None:
    if section is None:
        return

    raw_courses = section.pop(REGISTRY.keys.courses, None)
    raw_baseline_spec = section.pop(REGISTRY.keys.baseline, default_baseline_spec)

    if raw_courses is not None:
        if section.get(REGISTRY.keys.entries):
            raise ValueError("track_sampling.courses cannot be combined with entries")
        section[REGISTRY.keys.entries] = entries_from_courses(
            raw_courses,
            raw_baseline_spec,
            config_root=config_root,
            entry_from_course_variant=entry_from_course_variant,
        )
        return

    enrich_concrete_entries(section, config_root)


def expand_track_course_metadata(
    track: dict[str, object] | None,
    config_root: Path,
) -> None:
    if track is None:
        return
    enriched = enrich_track_with_registry_metadata(track, config_root=config_root)
    track.clear()
    track.update(enriched)
