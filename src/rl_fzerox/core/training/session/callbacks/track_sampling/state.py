# src/rl_fzerox/core/training/session/callbacks/track_sampling/state.py
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


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
    current_weight: float = 1.0

    def record_episode(
        self,
        frame_count: int,
        *,
        ema_alpha: float,
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
            return
        clamped_completion = max(0.0, min(1.0, float(completion_fraction)))
        if self.ema_completion_fraction is None:
            self.ema_completion_fraction = clamped_completion
            return
        self.ema_completion_fraction = (
            1.0 - ema_alpha
        ) * self.ema_completion_fraction + ema_alpha * clamped_completion


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
        )
    return tuple(grouped[course_key] for course_key in sorted(grouped))


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
