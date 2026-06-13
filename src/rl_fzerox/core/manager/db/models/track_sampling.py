# src/rl_fzerox/core/manager/db/models/track_sampling.py
"""ORM models for adaptive track sampling runtime state."""

from __future__ import annotations

from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped, mapped_column

from rl_fzerox.core.manager.db.models.base import ManagerBase


class RunTrackSamplingRuntimeModel(ManagerBase):
    """Global sampling settings and counters for one run."""

    __tablename__ = "run_track_sampling_runtime"

    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"), primary_key=True)
    sampling_mode: Mapped[str]
    action_repeat: Mapped[int]
    update_episodes: Mapped[int]
    ema_alpha: Mapped[float]
    max_weight_scale: Mapped[float]
    adaptive_completion_weight: Mapped[float]
    adaptive_target_completion: Mapped[float]
    adaptive_min_confidence_episodes: Mapped[int]
    adaptive_confidence_scale: Mapped[float]
    update_count: Mapped[int]
    episodes_since_update: Mapped[int]
    updated_at: Mapped[str]


class RunTrackSamplingEntryModel(ManagerBase):
    """Per-course sampling and X Cup generation state for one run."""

    __tablename__ = "run_track_sampling_entries"

    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"), primary_key=True)
    course_key: Mapped[str] = mapped_column(primary_key=True)
    track_id: Mapped[str]
    label: Mapped[str]
    base_weight: Mapped[float]
    current_weight: Mapped[float]
    completed_frames: Mapped[int]
    episode_count: Mapped[int]
    finished_episode_count: Mapped[int]
    success_sample_count: Mapped[int]
    ema_episode_frames: Mapped[float | None]
    ema_completion_fraction: Mapped[float | None]
    generation_episode_count: Mapped[int]
    generation_finished_episode_count: Mapped[int]
    generation_success_sample_count: Mapped[int]
    generation_ema_completion_fraction: Mapped[float | None]
    generated_course_slot: Mapped[int | None]
    generated_course_generation: Mapped[int | None]
    generated_course_id: Mapped[str | None]
    generated_course_name: Mapped[str | None]
    generated_course_hash: Mapped[str | None]
    generated_course_seed: Mapped[str | None]
    generated_course_segment_count: Mapped[int | None]
    generated_course_length: Mapped[float | None]


class RunTrackSamplingArtifactModel(ManagerBase):
    """Active materialized reset artifact for one sampled course variant."""

    __tablename__ = "run_track_sampling_artifacts"

    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"), primary_key=True)
    course_key: Mapped[str] = mapped_column(primary_key=True)
    reset_variant_key: Mapped[str] = mapped_column(primary_key=True)
    entry_id: Mapped[str]
    baseline_state_path: Mapped[str]
    metadata_path: Mapped[str]
    source_course_index: Mapped[int | None]
    source_gp_difficulty: Mapped[str | None]
    source_vehicle: Mapped[str | None]
    source_engine_setting_raw_value: Mapped[int | None]
    generated_course_slot: Mapped[int | None]
    generated_course_generation: Mapped[int | None]
    generated_course_id: Mapped[str | None]
    generated_course_name: Mapped[str | None]
    generated_course_hash: Mapped[str | None]
    generated_course_seed: Mapped[str | None]
    generated_course_segment_count: Mapped[int | None]
    generated_course_length: Mapped[float | None]


class RunTrackSamplingGeneratedSlotModel(ManagerBase):
    """Active generated X Cup identity for one stable sampling slot."""

    __tablename__ = "run_track_sampling_generated_slots"

    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"), primary_key=True)
    slot: Mapped[int] = mapped_column(primary_key=True)
    course_key: Mapped[str]
    generation: Mapped[int]
    course_id: Mapped[str]
    course_name: Mapped[str]
    course_hash: Mapped[str]
    course_seed: Mapped[str]
    segment_count: Mapped[int | None]
    course_length: Mapped[float | None]
    updated_at: Mapped[str]


class RunAltBaselineModel(ManagerBase):
    """User-created extra reset baseline for one stable course variant."""

    __tablename__ = "run_alt_baselines"

    id: Mapped[str] = mapped_column(primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"), index=True)
    course_key: Mapped[str] = mapped_column(index=True)
    reset_variant_key: Mapped[str]
    source_entry_id: Mapped[str]
    label: Mapped[str]
    state_path: Mapped[str]
    weight: Mapped[float]
    enabled: Mapped[bool]
    created_at: Mapped[str]
    updated_at: Mapped[str]
    deleted_at: Mapped[str | None] = mapped_column(default=None)
