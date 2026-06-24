# src/rl_fzerox/core/training/session/callbacks/track_sampling/state.py
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace


@dataclass(frozen=True, slots=True)
class TrackSamplingGenerationStats:
    episode_count: int = 0
    finished_episode_count: int = 0
    success_sample_count: int = 0
    ema_completion_fraction: float | None = None

    @classmethod
    def from_entry(cls, entry: TrackSamplingRuntimeEntry) -> TrackSamplingGenerationStats:
        return cls(
            episode_count=entry.generation_episode_count,
            finished_episode_count=entry.generation_finished_episode_count,
            success_sample_count=entry.generation_success_sample_count,
            ema_completion_fraction=entry.generation_ema_completion_fraction,
        )

    def merged(
        self,
        other: TrackSamplingGenerationStats,
    ) -> TrackSamplingGenerationStats:
        return TrackSamplingGenerationStats(
            episode_count=self.episode_count + other.episode_count,
            finished_episode_count=self.finished_episode_count + other.finished_episode_count,
            success_sample_count=self.success_sample_count + other.success_sample_count,
            ema_completion_fraction=_merged_generation_ema_completion_fraction(self, other),
        )

    def reset(self) -> TrackSamplingGenerationStats:
        return TrackSamplingGenerationStats()


@dataclass(frozen=True, slots=True)
class TrackSamplingGeneratedCourseMetadata:
    slot: int | None = None
    generation: int | None = None
    course_id: str | None = None
    name: str | None = None
    course_hash: str | None = None
    seed: int | None = None
    segment_count: int | None = None
    length: float | None = None

    @classmethod
    def from_entry(
        cls,
        entry: TrackSamplingRuntimeEntry,
    ) -> TrackSamplingGeneratedCourseMetadata:
        return cls(
            slot=entry.generated_course_slot,
            generation=entry.generated_course_generation,
            course_id=entry.generated_course_id,
            name=entry.generated_course_name,
            course_hash=entry.generated_course_hash,
            seed=entry.generated_course_seed,
            segment_count=entry.generated_course_segment_count,
            length=entry.generated_course_length,
        )

    def merged(
        self,
        other: TrackSamplingGeneratedCourseMetadata,
    ) -> TrackSamplingGeneratedCourseMetadata:
        return TrackSamplingGeneratedCourseMetadata(
            slot=_merged_optional_int(self.slot, other.slot),
            generation=_merged_optional_int(self.generation, other.generation),
            course_id=_merged_optional_str(self.course_id, other.course_id),
            name=_merged_optional_str(self.name, other.name),
            course_hash=_merged_optional_str(self.course_hash, other.course_hash),
            seed=_merged_optional_int(self.seed, other.seed),
            segment_count=_merged_optional_int(self.segment_count, other.segment_count),
            length=_merged_optional_float(self.length, other.length),
        )


@dataclass(frozen=True, slots=True)
class TrackSamplingRuntimeEntry:
    """Persisted per-course sampler stats.

    episode_count counts terminal samples with usable frame counts.
    finished_episode_count is the subset that finished the race.
    success_sample_count means the sampler saw a valid episode sample, not that
    the race was won or completed. completion_sample_count only counts samples
    with a progress fraction, so it may lag episode_count. generation_* counters
    are scoped to the current generated X Cup course and reset on rotation.
    """

    track_id: str
    course_key: str
    label: str
    base_weight: float
    current_weight: float
    completed_frames: int
    episode_count: int
    finished_episode_count: int
    success_sample_count: int
    ema_episode_frames: float | None
    ema_completion_fraction: float | None
    completion_sample_count: int = 0
    completion_fraction_total: float = 0.0
    ema_finish_rate: float | None = None
    current_problem_score: float = 0.0
    generation_episode_count: int = 0
    generation_finished_episode_count: int = 0
    generation_success_sample_count: int = 0
    generation_ema_completion_fraction: float | None = None
    generated_course_slot: int | None = None
    generated_course_generation: int | None = None
    generated_course_id: str | None = None
    generated_course_name: str | None = None
    generated_course_hash: str | None = None
    generated_course_seed: int | None = None
    generated_course_segment_count: int | None = None
    generated_course_length: float | None = None

    @property
    def generation_stats(self) -> TrackSamplingGenerationStats:
        return TrackSamplingGenerationStats.from_entry(self)

    @property
    def generated_course(self) -> TrackSamplingGeneratedCourseMetadata:
        return TrackSamplingGeneratedCourseMetadata.from_entry(self)


@dataclass(frozen=True, slots=True)
class DeficitBudgetCourseSchedulerState:
    course_key: str
    uniform_deficit_steps: float = 0.0
    adaptive_deficit_steps: float = 0.0
    scheduler_env_steps: int = 0
    last_uniform_assignment_index: int = 0


@dataclass(frozen=True, slots=True)
class DeficitBudgetSchedulerState:
    uniform_lane_deficit_steps: float = 0.0
    adaptive_lane_deficit_steps: float = 0.0
    uniform_assignment_count: int = 0
    entries: tuple[DeficitBudgetCourseSchedulerState, ...] = ()


@dataclass(frozen=True, slots=True)
class TrackSamplingRuntimeState:
    sampling_mode: str
    action_repeat: int
    update_episodes: int
    ema_alpha: float
    max_weight_scale: float
    adaptive_completion_weight: float
    adaptive_target_completion: float
    adaptive_min_confidence_episodes: int
    adaptive_confidence_scale: float
    update_count: int
    episodes_since_update: int
    entries: tuple[TrackSamplingRuntimeEntry, ...]
    deficit_budget_difficulty_metric: str = "completion_ema"
    deficit_budget_warmup_min_episodes_per_course: int = 0
    deficit_budget_scheduler: DeficitBudgetSchedulerState | None = None


@dataclass(slots=True)
class TrackStepStats:
    """Mutable training-time mirror of TrackSamplingRuntimeEntry counters."""

    base_weight: float
    completed_frames: int = 0
    episode_count: int = 0
    finished_episode_count: int = 0
    success_sample_count: int = 0
    completion_sample_count: int = 0
    completion_fraction_total: float = 0.0
    ema_episode_frames: float | None = None
    ema_completion_fraction: float | None = None
    ema_finish_rate: float | None = None
    current_problem_score: float = 0.0
    generation_episode_count: int = 0
    generation_finished_episode_count: int = 0
    generation_success_sample_count: int = 0
    generation_ema_completion_fraction: float | None = None
    current_weight: float = 1.0

    def record_episode(
        self,
        frame_count: int,
        *,
        ema_alpha: float,
        generation_ema_alpha: float | None = None,
        completion_fraction: float | None,
        finished: bool,
    ) -> None:
        self.completed_frames += frame_count
        self.episode_count += 1
        self.success_sample_count += 1
        if finished:
            self.finished_episode_count += 1
        finish_value = 1.0 if finished else 0.0
        if self.ema_finish_rate is None:
            self.ema_finish_rate = finish_value
        else:
            self.ema_finish_rate = (1.0 - ema_alpha) * self.ema_finish_rate + (
                ema_alpha * finish_value
            )
        if self.ema_episode_frames is None:
            self.ema_episode_frames = float(frame_count)
        else:
            self.ema_episode_frames = (
                1.0 - ema_alpha
            ) * self.ema_episode_frames + ema_alpha * frame_count
        if completion_fraction is None:
            if generation_ema_alpha is not None:
                self._record_generation_episode(
                    completion_fraction=None,
                    finished=finished,
                    ema_alpha=generation_ema_alpha,
                )
            return
        clamped_completion = max(0.0, min(1.0, float(completion_fraction)))
        self.completion_sample_count += 1
        self.completion_fraction_total += clamped_completion
        if self.ema_completion_fraction is None:
            self.ema_completion_fraction = clamped_completion
        else:
            self.ema_completion_fraction = (
                1.0 - ema_alpha
            ) * self.ema_completion_fraction + ema_alpha * clamped_completion
        if generation_ema_alpha is not None:
            self._record_generation_episode(
                completion_fraction=clamped_completion,
                finished=finished,
                ema_alpha=generation_ema_alpha,
            )

    def _record_generation_episode(
        self,
        *,
        completion_fraction: float | None,
        finished: bool,
        ema_alpha: float,
    ) -> None:
        self.generation_episode_count += 1
        self.generation_success_sample_count += 1
        if finished:
            self.generation_finished_episode_count += 1
        if completion_fraction is None:
            return
        if self.generation_ema_completion_fraction is None:
            self.generation_ema_completion_fraction = completion_fraction
            return
        self.generation_ema_completion_fraction = (
            1.0 - ema_alpha
        ) * self.generation_ema_completion_fraction + ema_alpha * completion_fraction


def aggregate_runtime_entries(
    entries: Sequence[TrackSamplingRuntimeEntry],
) -> tuple[TrackSamplingRuntimeEntry, ...]:
    grouped: dict[str, TrackSamplingRuntimeEntry] = {}
    for entry in entries:
        course_key = entry.course_key
        existing = grouped.get(course_key)
        if existing is None:
            grouped[course_key] = replace(entry, track_id=course_key, course_key=course_key)
            continue
        generation_stats = existing.generation_stats.merged(entry.generation_stats)
        generated_course = existing.generated_course.merged(entry.generated_course)
        grouped[course_key] = TrackSamplingRuntimeEntry(
            track_id=course_key,
            course_key=course_key,
            label=existing.label,
            base_weight=existing.base_weight + entry.base_weight,
            current_weight=existing.current_weight + entry.current_weight,
            completed_frames=existing.completed_frames + entry.completed_frames,
            episode_count=existing.episode_count + entry.episode_count,
            finished_episode_count=(existing.finished_episode_count + entry.finished_episode_count),
            success_sample_count=existing.success_sample_count + entry.success_sample_count,
            ema_episode_frames=_merged_ema_episode_frames(existing, entry),
            ema_completion_fraction=_merged_ema_completion_fraction(existing, entry),
            completion_sample_count=(
                existing.completion_sample_count + entry.completion_sample_count
            ),
            completion_fraction_total=(
                existing.completion_fraction_total + entry.completion_fraction_total
            ),
            ema_finish_rate=_merged_ema_finish_rate(existing, entry),
            current_problem_score=_merged_problem_score(existing, entry),
            generation_episode_count=generation_stats.episode_count,
            generation_finished_episode_count=generation_stats.finished_episode_count,
            generation_success_sample_count=generation_stats.success_sample_count,
            generation_ema_completion_fraction=generation_stats.ema_completion_fraction,
            generated_course_slot=generated_course.slot,
            generated_course_generation=generated_course.generation,
            generated_course_id=generated_course.course_id,
            generated_course_name=generated_course.name,
            generated_course_hash=generated_course.course_hash,
            generated_course_seed=generated_course.seed,
            generated_course_segment_count=generated_course.segment_count,
            generated_course_length=generated_course.length,
        )
    return tuple(grouped[course_key] for course_key in sorted(grouped))


def replace_runtime_generation(
    state: TrackSamplingRuntimeState,
    *,
    course_key: str,
    replacement_label: str,
    generated_course_slot: int | None,
    generated_course_generation: int | None,
    generated_course_id: str | None,
    generated_course_name: str | None,
    generated_course_hash: str | None,
    generated_course_seed: int | None,
    generated_course_segment_count: int | None,
    generated_course_length: float | None,
) -> TrackSamplingRuntimeState:
    """Carry slot sampling history forward while resetting current-course stats."""

    generated_course = TrackSamplingGeneratedCourseMetadata(
        slot=generated_course_slot,
        generation=generated_course_generation,
        course_id=generated_course_id,
        name=generated_course_name,
        course_hash=generated_course_hash,
        seed=generated_course_seed,
        segment_count=generated_course_segment_count,
        length=generated_course_length,
    )
    entries = tuple(
        _replacement_runtime_entry(
            entry,
            replacement_label=replacement_label,
            generated_course=generated_course,
        )
        if entry.course_key == course_key
        else entry
        for entry in state.entries
    )
    return replace(state, entries=entries)


def _replacement_runtime_entry(
    entry: TrackSamplingRuntimeEntry,
    *,
    replacement_label: str,
    generated_course: TrackSamplingGeneratedCourseMetadata,
) -> TrackSamplingRuntimeEntry:
    generation_stats = entry.generation_stats.reset()
    return replace(
        entry,
        label=replacement_label,
        generation_episode_count=generation_stats.episode_count,
        generation_finished_episode_count=generation_stats.finished_episode_count,
        generation_success_sample_count=generation_stats.success_sample_count,
        generation_ema_completion_fraction=generation_stats.ema_completion_fraction,
        generated_course_slot=generated_course.slot,
        generated_course_generation=generated_course.generation,
        generated_course_id=generated_course.course_id,
        generated_course_name=generated_course.name,
        generated_course_hash=generated_course.course_hash,
        generated_course_seed=generated_course.seed,
        generated_course_segment_count=generated_course.segment_count,
        generated_course_length=generated_course.length,
    )


def _merged_ema_episode_frames(
    left: TrackSamplingRuntimeEntry,
    right: TrackSamplingRuntimeEntry,
) -> float | None:
    if left.ema_episode_frames is None:
        return right.ema_episode_frames
    if right.ema_episode_frames is None:
        return left.ema_episode_frames
    left_weight = max(left.episode_count, 1)
    right_weight = max(right.episode_count, 1)
    total_weight = left_weight + right_weight
    if total_weight <= 0:
        return None
    return (
        left.ema_episode_frames * left_weight + right.ema_episode_frames * right_weight
    ) / total_weight


def _merged_ema_completion_fraction(
    left: TrackSamplingRuntimeEntry,
    right: TrackSamplingRuntimeEntry,
) -> float | None:
    if left.ema_completion_fraction is None:
        return right.ema_completion_fraction
    if right.ema_completion_fraction is None:
        return left.ema_completion_fraction
    left_weight = max(left.episode_count, 1)
    right_weight = max(right.episode_count, 1)
    total_weight = left_weight + right_weight
    if total_weight <= 0:
        return None
    return (
        left.ema_completion_fraction * left_weight + right.ema_completion_fraction * right_weight
    ) / total_weight


def _merged_ema_finish_rate(
    left: TrackSamplingRuntimeEntry,
    right: TrackSamplingRuntimeEntry,
) -> float | None:
    if left.ema_finish_rate is None:
        return right.ema_finish_rate
    if right.ema_finish_rate is None:
        return left.ema_finish_rate
    left_weight = max(left.episode_count, 1)
    right_weight = max(right.episode_count, 1)
    total_weight = left_weight + right_weight
    if total_weight <= 0:
        return None
    return (
        left.ema_finish_rate * left_weight + right.ema_finish_rate * right_weight
    ) / total_weight


def _merged_problem_score(
    left: TrackSamplingRuntimeEntry,
    right: TrackSamplingRuntimeEntry,
) -> float:
    left_weight = max(left.episode_count, 1)
    right_weight = max(right.episode_count, 1)
    total_weight = left_weight + right_weight
    if total_weight <= 0:
        return 0.0
    return (
        left.current_problem_score * left_weight + right.current_problem_score * right_weight
    ) / total_weight


def _merged_generation_ema_completion_fraction(
    left: TrackSamplingGenerationStats,
    right: TrackSamplingGenerationStats,
) -> float | None:
    if left.ema_completion_fraction is None:
        return right.ema_completion_fraction
    if right.ema_completion_fraction is None:
        return left.ema_completion_fraction
    left_weight = max(left.episode_count, 1)
    right_weight = max(right.episode_count, 1)
    total_weight = left_weight + right_weight
    if total_weight <= 0:
        return None
    return (
        left.ema_completion_fraction * left_weight + right.ema_completion_fraction * right_weight
    ) / total_weight


def _merged_optional_int(left: int | None, right: int | None) -> int | None:
    if left is None:
        return right
    if right is None or right == left:
        return left
    return None


def _merged_optional_float(left: float | None, right: float | None) -> float | None:
    if left is None:
        return right
    if right is None or right == left:
        return left
    return None


def _merged_optional_str(left: str | None, right: str | None) -> str | None:
    if left is None:
        return right
    if right is None or right == left:
        return left
    return None
