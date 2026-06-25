# src/rl_fzerox/core/runtime_spec/schema/tracks/config.py
"""Concrete track metadata runtime schema.

``TrackConfig`` describes the resolved track/mode/vehicle metadata attached to
a runtime app config after manager projection and registry expansion. Sampling
entries reuse the same domain vocabulary but live in ``sampling.py`` because
reset selection has additional scheduling fields and generated-course metadata.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, NonNegativeInt

from rl_fzerox.core.domain.race import RaceDifficultyName
from rl_fzerox.core.runtime_spec.schema.tracks.records import TrackRecordsConfig


class TrackConfig(BaseModel):
    """Metadata for a concrete track/mode baseline."""

    model_config = ConfigDict(extra="forbid")

    id: str | None = None
    display_name: str | None = None
    course_ref: str | None = None
    course_id: str | None = None
    course_name: str | None = None
    course_index: NonNegativeInt | None = None
    mode: str | None = None
    gp_difficulty: RaceDifficultyName | None = None
    vehicle: str | None = None
    vehicle_name: str | None = None
    source_vehicle: str | None = None
    engine_setting_raw_value: NonNegativeInt | None = None
    engine_setting_min_raw_value: NonNegativeInt | None = None
    engine_setting_max_raw_value: NonNegativeInt | None = None
    source_course_index: NonNegativeInt | None = None
    source_gp_difficulty: RaceDifficultyName | None = None
    source_engine_setting_raw_value: NonNegativeInt | None = None
    baseline_state_path: Path | None = None
    records: TrackRecordsConfig | None = None
    notes: str | None = None
