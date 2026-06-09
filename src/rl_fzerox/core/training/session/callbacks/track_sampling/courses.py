# src/rl_fzerox/core/training/session/callbacks/track_sampling/courses.py
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TypeVar

from rl_fzerox.core.runtime_spec.schema import TrackSamplingConfig, TrackSamplingEntryConfig
from rl_fzerox.core.training.session.callbacks.track_sampling.episodes import sanitize_log_key
from rl_fzerox.core.training.session.callbacks.track_sampling.state import (
    TrackSamplingRuntimeEntry,
    TrackStepStats,
)

_ValueT = TypeVar("_ValueT")


@dataclass(frozen=True, slots=True)
class GeneratedCourseMetadata:
    slot: int | None = None
    generation: int | None = None
    course_id: str | None = None
    name: str | None = None
    hash: str | None = None
    seed: int | None = None
    segment_count: int | None = None
    length: float | None = None

    @classmethod
    def from_entry(cls, entry: TrackSamplingEntryConfig) -> GeneratedCourseMetadata:
        if entry.generated_course_slot is None and entry.generated_course_kind is None:
            return cls()
        return cls(
            slot=_optional_int(entry.generated_course_slot),
            generation=_optional_int(entry.generated_course_generation),
            course_id=entry.course_id,
            name=entry.course_name or entry.display_name or entry.course_id or entry.id,
            hash=entry.generated_course_hash,
            seed=_optional_int(entry.generated_course_seed),
            segment_count=_optional_int(entry.generated_course_segment_count),
            length=_optional_float(entry.generated_course_length),
        )

    @classmethod
    def from_runtime_entry(
        cls,
        entry: TrackSamplingRuntimeEntry,
    ) -> GeneratedCourseMetadata:
        return cls(
            slot=entry.generated_course_slot,
            generation=entry.generated_course_generation,
            course_id=entry.generated_course_id,
            name=entry.generated_course_name,
            hash=entry.generated_course_hash,
            seed=entry.generated_course_seed,
            segment_count=entry.generated_course_segment_count,
            length=entry.generated_course_length,
        )

    def overlay(self, other: GeneratedCourseMetadata) -> GeneratedCourseMetadata:
        return GeneratedCourseMetadata(
            slot=_overlay_value(self.slot, other.slot),
            generation=_overlay_value(self.generation, other.generation),
            course_id=_overlay_value(self.course_id, other.course_id),
            name=_overlay_value(self.name, other.name),
            hash=_overlay_value(self.hash, other.hash),
            seed=_overlay_value(self.seed, other.seed),
            segment_count=_overlay_value(self.segment_count, other.segment_count),
            length=_overlay_value(self.length, other.length),
        )

    def fill_missing(self, other: GeneratedCourseMetadata) -> GeneratedCourseMetadata:
        return GeneratedCourseMetadata(
            slot=_overlay_value(other.slot, self.slot),
            generation=_overlay_value(other.generation, self.generation),
            course_id=_overlay_value(other.course_id, self.course_id),
            name=_overlay_value(other.name, self.name),
            hash=_overlay_value(other.hash, self.hash),
            seed=_overlay_value(other.seed, self.seed),
            segment_count=_overlay_value(other.segment_count, self.segment_count),
            length=_overlay_value(other.length, self.length),
        )


@dataclass(frozen=True, slots=True)
class TrackSamplingCourseEntry:
    entry_id: str
    base_weight: float
    course_key: str | None = None
    label: str | None = None
    log_key: str | None = None
    log_enabled: bool = True
    generated: GeneratedCourseMetadata = field(default_factory=GeneratedCourseMetadata)


@dataclass(frozen=True, slots=True)
class ResolvedTrackSamplingCourse:
    course_key: str
    entry_ids: tuple[str, ...]
    label: str
    log_key: str
    log_enabled: bool
    base_weight_total: float
    base_weight_mean: float
    generated: GeneratedCourseMetadata

    def with_runtime_generation(
        self,
        entry: TrackSamplingRuntimeEntry,
    ) -> ResolvedTrackSamplingCourse:
        return replace(
            self,
            generated=self.generated.overlay(GeneratedCourseMetadata.from_runtime_entry(entry)),
        )

    def runtime_entry(
        self,
        *,
        stats: TrackStepStats,
        completed_frames: int | None = None,
    ) -> TrackSamplingRuntimeEntry:
        generated = self.generated
        return TrackSamplingRuntimeEntry(
            track_id=self.course_key,
            course_key=self.course_key,
            label=self.label,
            base_weight=stats.base_weight,
            current_weight=stats.current_weight,
            completed_frames=(
                stats.completed_frames if completed_frames is None else completed_frames
            ),
            episode_count=stats.episode_count,
            finished_episode_count=stats.finished_episode_count,
            success_sample_count=stats.success_sample_count,
            ema_episode_frames=stats.ema_episode_frames,
            ema_completion_fraction=stats.ema_completion_fraction,
            generation_episode_count=stats.generation_episode_count,
            generation_finished_episode_count=stats.generation_finished_episode_count,
            generation_success_sample_count=stats.generation_success_sample_count,
            generation_ema_completion_fraction=stats.generation_ema_completion_fraction,
            generated_course_slot=generated.slot,
            generated_course_generation=generated.generation,
            generated_course_id=generated.course_id,
            generated_course_name=generated.name,
            generated_course_hash=generated.hash,
            generated_course_seed=generated.seed,
            generated_course_segment_count=generated.segment_count,
            generated_course_length=generated.length,
        )


@dataclass(frozen=True, slots=True)
class ResolvedTrackSamplingCourses:
    entry_base_weights: dict[str, float]
    entry_course_keys: dict[str, str]
    courses: dict[str, ResolvedTrackSamplingCourse]

    @property
    def course_entry_ids(self) -> dict[str, tuple[str, ...]]:
        return {course_key: course.entry_ids for course_key, course in self.courses.items()}

    @property
    def course_entry_base_totals(self) -> dict[str, float]:
        return {course_key: course.base_weight_total for course_key, course in self.courses.items()}


def resolve_track_sampling_courses_from_configs(
    configs: Sequence[TrackSamplingConfig],
) -> ResolvedTrackSamplingCourses:
    entries: list[TrackSamplingCourseEntry] = []
    for config in configs:
        for entry in config.entries:
            course_key = _entry_course_key(entry)
            entries.append(
                TrackSamplingCourseEntry(
                    entry_id=entry.id,
                    base_weight=float(entry.weight),
                    course_key=course_key,
                    label=entry.course_name or entry.course_id or entry.display_name or entry.id,
                    log_key=course_key,
                    log_enabled=entry.log_per_course,
                    generated=GeneratedCourseMetadata.from_entry(entry),
                )
            )
    return resolve_track_sampling_courses_from_entries(entries)


def resolve_track_sampling_courses_from_entries(
    entries: Sequence[TrackSamplingCourseEntry],
) -> ResolvedTrackSamplingCourses:
    base_weights: dict[str, float] = {}
    builders: dict[str, _CourseBuilder] = {}
    entry_course_keys: dict[str, str] = {}
    for entry in entries:
        entry_id = entry.entry_id
        base_weights.setdefault(entry_id, float(entry.base_weight))
        course_key = entry.course_key or entry_id
        entry_course_keys.setdefault(entry_id, course_key)
        builder = builders.setdefault(
            course_key,
            _CourseBuilder(
                course_key=course_key,
                label=entry.label or course_key,
                log_key=sanitize_log_key(entry.log_key or course_key),
                log_enabled=False,
            ),
        )
        builder.add_entry(
            entry_id=entry_id,
            base_weight=base_weights[entry_id],
            log_enabled=entry.log_enabled,
            generated=entry.generated,
        )
    return _resolved_courses(
        entry_base_weights=base_weights,
        entry_course_keys=entry_course_keys,
        builders=builders,
    )


@dataclass(slots=True)
class _CourseBuilder:
    course_key: str
    label: str
    log_key: str
    log_enabled: bool
    entry_ids: list[str] | None = None
    base_weight_total: float = 0.0
    generated: GeneratedCourseMetadata = field(default_factory=GeneratedCourseMetadata)

    def add_entry(
        self,
        *,
        entry_id: str,
        base_weight: float,
        log_enabled: bool,
        generated: GeneratedCourseMetadata,
    ) -> None:
        if self.entry_ids is None:
            self.entry_ids = []
        self.entry_ids.append(entry_id)
        self.base_weight_total += base_weight
        self.log_enabled = self.log_enabled or log_enabled
        self.generated = self.generated.fill_missing(generated)

    def build(self) -> ResolvedTrackSamplingCourse:
        entry_ids = tuple(self.entry_ids or ())
        return ResolvedTrackSamplingCourse(
            course_key=self.course_key,
            entry_ids=entry_ids,
            label=self.label,
            log_key=self.log_key,
            log_enabled=self.log_enabled,
            base_weight_total=self.base_weight_total,
            base_weight_mean=self.base_weight_total / max(len(entry_ids), 1),
            generated=self.generated,
        )


def _resolved_courses(
    *,
    entry_base_weights: dict[str, float],
    entry_course_keys: dict[str, str],
    builders: Mapping[str, _CourseBuilder],
) -> ResolvedTrackSamplingCourses:
    return ResolvedTrackSamplingCourses(
        entry_base_weights=entry_base_weights,
        entry_course_keys=entry_course_keys,
        courses={course_key: builder.build() for course_key, builder in builders.items()},
    )


def _entry_course_key(entry: TrackSamplingEntryConfig) -> str:
    return entry.runtime_course_key or entry.course_id or entry.id


def _overlay_value(old: _ValueT | None, new: _ValueT | None) -> _ValueT | None:
    return old if new is None else new


def _optional_int(value: int | None) -> int | None:
    return None if value is None else int(value)


def _optional_float(value: float | None) -> float | None:
    return None if value is None else float(value)
