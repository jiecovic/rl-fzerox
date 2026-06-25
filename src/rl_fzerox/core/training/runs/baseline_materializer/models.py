# src/rl_fzerox/core/training/runs/baseline_materializer/models.py
"""Data contracts for baseline materialization requests and artifacts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from fzerox_emulator import EmulatorBackend, ResetState
from rl_fzerox.core.domain.courses import XCupGeneratedCourseKind
from rl_fzerox.core.domain.race import RaceDifficultyName

BaselineArtifactSource = Literal["existing", "cache", "generated"]


@dataclass(frozen=True, slots=True)
class BaselineRequest:
    """Resolved input needed to generate one run-local reset state."""

    label: str
    source_state_path: Path | None = None
    course_id: str | None = None
    course_name: str | None = None
    course_index: int | None = None
    mode: str | None = None
    gp_difficulty: RaceDifficultyName | None = None
    vehicle: str | None = None
    vehicle_name: str | None = None
    source_vehicle: str | None = None
    engine_setting_raw_value: int | None = None
    source_course_index: int | None = None
    source_gp_difficulty: RaceDifficultyName | None = None
    source_engine_setting_raw_value: int | None = None
    camera_setting: str | None = None
    baseline_variant_index: int | None = None
    baseline_variant_count: int | None = None
    baseline_variant_seed: int | None = None
    generated_course_kind: XCupGeneratedCourseKind | None = None
    generated_course_seed: int | None = None
    generated_course_hash: str | None = None
    generated_course_slot: int | None = None
    generated_course_generation: int | None = None


@dataclass(frozen=True, slots=True)
class BaselineArtifact:
    """Run-local materialized baseline artifact and metadata paths."""

    state_path: Path
    metadata_path: Path
    cache_key: str
    source: BaselineArtifactSource
    source_course_index: int | None = None
    source_vehicle: str | None = None
    source_gp_difficulty: RaceDifficultyName | None = None
    source_engine_setting_raw_value: int | None = None
    generated_course_segment_count: int | None = None
    generated_course_length: float | None = None


@dataclass(frozen=True, slots=True)
class BaselineMaterializerContext:
    """Runtime dependencies needed when a materialized state must be generated."""

    core_path: Path
    rom_path: Path
    renderer: str
    race_intro_target_timer: int | None
    core_sha256: str
    rom_sha256: str


RaceStartMaterializer = Callable[..., None]
GenericModeSeedMaterializer = Callable[..., None]


class StateSavingEmulator(EmulatorBackend, Protocol):
    def reset(self) -> ResetState: ...

    def save_state(self, path: Path) -> None: ...

    def close(self) -> None: ...
