# src/rl_fzerox/core/training/session/callbacks/track_sampling/state.py
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace


@dataclass(frozen=True, slots=True)
class TrackSamplingRuntimeEntry:
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
    generation_episode_count: int = 0
    generation_finished_episode_count: int = 0
    generation_success_sample_count: int = 0
    generation_ema_completion_fraction: float | None = None
    generated_course_slot: int | None = None
    generated_course_generation: int | None = None
    generated_entry_id: str | None = None
    generated_course_id: str | None = None
    generated_course_name: str | None = None
    generated_course_hash: str | None = None
    generated_course_seed: int | None = None
    generated_baseline_state_path: str | None = None
    generated_course_segment_count: int | None = None
    generated_course_length: float | None = None


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


@dataclass(slots=True)
class TrackStepStats:
    base_weight: float
    completed_frames: int = 0
    episode_count: int = 0
    finished_episode_count: int = 0
    success_sample_count: int = 0
    ema_episode_frames: float | None = None
    ema_completion_fraction: float | None = None
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
            grouped[course_key] = TrackSamplingRuntimeEntry(
                track_id=course_key,
                course_key=course_key,
                label=entry.label,
                base_weight=entry.base_weight,
                current_weight=entry.current_weight,
                completed_frames=entry.completed_frames,
                episode_count=entry.episode_count,
                finished_episode_count=entry.finished_episode_count,
                success_sample_count=entry.success_sample_count,
                ema_episode_frames=entry.ema_episode_frames,
                ema_completion_fraction=entry.ema_completion_fraction,
                generation_episode_count=entry.generation_episode_count,
                generation_finished_episode_count=entry.generation_finished_episode_count,
                generation_success_sample_count=entry.generation_success_sample_count,
                generation_ema_completion_fraction=entry.generation_ema_completion_fraction,
                generated_course_slot=entry.generated_course_slot,
                generated_course_generation=entry.generated_course_generation,
                generated_entry_id=entry.generated_entry_id,
                generated_course_id=entry.generated_course_id,
                generated_course_name=entry.generated_course_name,
                generated_course_hash=entry.generated_course_hash,
                generated_course_seed=entry.generated_course_seed,
                generated_baseline_state_path=entry.generated_baseline_state_path,
                generated_course_segment_count=entry.generated_course_segment_count,
                generated_course_length=entry.generated_course_length,
            )
            continue
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
            generation_episode_count=(
                existing.generation_episode_count + entry.generation_episode_count
            ),
            generation_finished_episode_count=(
                existing.generation_finished_episode_count + entry.generation_finished_episode_count
            ),
            generation_success_sample_count=(
                existing.generation_success_sample_count + entry.generation_success_sample_count
            ),
            generation_ema_completion_fraction=_merged_generation_ema_completion_fraction(
                existing,
                entry,
            ),
            generated_course_slot=_merged_optional_int(
                existing.generated_course_slot,
                entry.generated_course_slot,
            ),
            generated_course_generation=_merged_optional_int(
                existing.generated_course_generation,
                entry.generated_course_generation,
            ),
            generated_entry_id=_merged_optional_str(
                existing.generated_entry_id,
                entry.generated_entry_id,
            ),
            generated_course_id=_merged_optional_str(
                existing.generated_course_id,
                entry.generated_course_id,
            ),
            generated_course_name=_merged_optional_str(
                existing.generated_course_name,
                entry.generated_course_name,
            ),
            generated_course_hash=_merged_optional_str(
                existing.generated_course_hash,
                entry.generated_course_hash,
            ),
            generated_course_seed=_merged_optional_int(
                existing.generated_course_seed,
                entry.generated_course_seed,
            ),
            generated_baseline_state_path=_merged_optional_str(
                existing.generated_baseline_state_path,
                entry.generated_baseline_state_path,
            ),
            generated_course_segment_count=_merged_optional_int(
                existing.generated_course_segment_count,
                entry.generated_course_segment_count,
            ),
            generated_course_length=_merged_optional_float(
                existing.generated_course_length,
                entry.generated_course_length,
            ),
        )
    return tuple(grouped[course_key] for course_key in sorted(grouped))


def replace_runtime_generation(
    state: TrackSamplingRuntimeState,
    *,
    course_key: str,
    replacement_label: str,
    generated_course_slot: int | None,
    generated_course_generation: int | None,
    generated_entry_id: str | None,
    generated_course_id: str | None,
    generated_course_name: str | None,
    generated_course_hash: str | None,
    generated_course_seed: int | None,
    generated_baseline_state_path: str | None,
    generated_course_segment_count: int | None,
    generated_course_length: float | None,
) -> TrackSamplingRuntimeState:
    """Carry slot sampling history forward while resetting current-course stats."""

    entries = tuple(
        _replacement_runtime_entry(
            entry,
            replacement_label=replacement_label,
            generated_course_slot=generated_course_slot,
            generated_course_generation=generated_course_generation,
            generated_entry_id=generated_entry_id,
            generated_course_id=generated_course_id,
            generated_course_name=generated_course_name,
            generated_course_hash=generated_course_hash,
            generated_course_seed=generated_course_seed,
            generated_baseline_state_path=generated_baseline_state_path,
            generated_course_segment_count=generated_course_segment_count,
            generated_course_length=generated_course_length,
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
    generated_course_slot: int | None,
    generated_course_generation: int | None,
    generated_entry_id: str | None,
    generated_course_id: str | None,
    generated_course_name: str | None,
    generated_course_hash: str | None,
    generated_course_seed: int | None,
    generated_baseline_state_path: str | None,
    generated_course_segment_count: int | None,
    generated_course_length: float | None,
) -> TrackSamplingRuntimeEntry:
    return replace(
        entry,
        label=replacement_label,
        generation_episode_count=0,
        generation_finished_episode_count=0,
        generation_success_sample_count=0,
        generation_ema_completion_fraction=None,
        generated_course_slot=generated_course_slot,
        generated_course_generation=generated_course_generation,
        generated_entry_id=generated_entry_id,
        generated_course_id=generated_course_id,
        generated_course_name=generated_course_name,
        generated_course_hash=generated_course_hash,
        generated_course_seed=generated_course_seed,
        generated_baseline_state_path=generated_baseline_state_path,
        generated_course_segment_count=generated_course_segment_count,
        generated_course_length=generated_course_length,
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


def _merged_generation_ema_completion_fraction(
    left: TrackSamplingRuntimeEntry,
    right: TrackSamplingRuntimeEntry,
) -> float | None:
    if left.generation_ema_completion_fraction is None:
        return right.generation_ema_completion_fraction
    if right.generation_ema_completion_fraction is None:
        return left.generation_ema_completion_fraction
    left_weight = max(left.generation_episode_count, 1)
    right_weight = max(right.generation_episode_count, 1)
    total_weight = left_weight + right_weight
    if total_weight <= 0:
        return None
    return (
        left.generation_ema_completion_fraction * left_weight
        + right.generation_ema_completion_fraction * right_weight
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
