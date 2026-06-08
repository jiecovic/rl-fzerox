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
