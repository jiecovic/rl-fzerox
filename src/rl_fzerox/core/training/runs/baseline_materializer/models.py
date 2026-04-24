# src/rl_fzerox/core/training/runs/baseline_materializer/models.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class BaselineRequest:
    """Resolved input needed to generate one run-local reset state."""

    label: str
    source_state_path: Path | None = None
    course_id: str | None = None
    course_name: str | None = None
    course_index: int | None = None
    mode: str | None = None
    vehicle: str | None = None
    vehicle_name: str | None = None
    source_vehicle: str | None = None
    engine_setting: str | None = None
    engine_setting_raw_value: int | None = None
    source_course_index: int | None = None
    source_engine_setting: str | None = None
    source_engine_setting_raw_value: int | None = None
    camera_setting: str | None = None


@dataclass(frozen=True, slots=True)
class BaselineArtifact:
    """Run-local materialized baseline artifact and metadata paths."""

    state_path: Path
    metadata_path: Path
    cache_key: str


@dataclass(frozen=True, slots=True)
class BaselineMaterializerContext:
    """Runtime dependencies needed when a materialized state must be generated."""

    core_path: Path
    rom_path: Path
    renderer: str
    race_intro_target_timer: int | None
