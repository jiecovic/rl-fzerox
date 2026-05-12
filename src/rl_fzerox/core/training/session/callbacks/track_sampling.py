# src/rl_fzerox/core/training/session/callbacks/track_sampling.py
from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rl_fzerox.core.runtime_spec.schema import CurriculumConfig, EnvConfig, TrackSamplingConfig


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
    update_count: int
    episodes_since_update: int
    entries: tuple[TrackSamplingRuntimeEntry, ...]


@dataclass(slots=True)
class _TrackStepStats:
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
            (1.0 - ema_alpha) * self.ema_completion_fraction + ema_alpha * clamped_completion
        )


class StepBalancedTrackSamplingController:
    """Adapt track reset weights toward equal frame coverage with optional completion bias."""

    def __init__(
        self,
        *,
        track_base_weights: dict[str, float],
        sampling_mode: str = "step_balanced",
        action_repeat: int,
        update_episodes: int,
        ema_alpha: float,
        max_weight_scale: float,
        adaptive_completion_weight: float = 0.35,
        adaptive_target_completion: float = 0.9,
        log_details: bool = False,
        track_course_keys: dict[str, str] | None = None,
        track_log_keys: dict[str, str] | None = None,
        track_labels: dict[str, str] | None = None,
        restored_state: TrackSamplingRuntimeState | None = None,
    ) -> None:
        self._entry_base_weights = {
            track_id: float(weight) for track_id, weight in track_base_weights.items()
        }
        course_keys = {
            track_id: (
                (track_course_keys or {}).get(track_id)
                or (track_log_keys or {}).get(track_id)
                or track_id
            )
            for track_id in self._entry_base_weights
        }
        self._entry_course_keys = course_keys
        course_entry_ids: dict[str, list[str]] = {}
        course_labels: dict[str, str] = {}
        course_log_keys: dict[str, str] = {}
        course_base_weight_sums: dict[str, float] = {}
        course_base_weight_counts: dict[str, int] = {}

        for track_id, base_weight in self._entry_base_weights.items():
            course_key = course_keys[track_id]
            course_entry_ids.setdefault(course_key, []).append(track_id)
            course_labels.setdefault(
                course_key,
                (track_labels or {}).get(track_id, course_key),
            )
            course_log_keys.setdefault(
                course_key,
                _sanitize_log_key((track_log_keys or {}).get(track_id, course_key)),
            )
            course_base_weight_sums[course_key] = (
                course_base_weight_sums.get(course_key, 0.0) + base_weight
            )
            course_base_weight_counts[course_key] = course_base_weight_counts.get(course_key, 0) + 1

        self._course_entry_ids = {
            course_key: tuple(entry_ids) for course_key, entry_ids in course_entry_ids.items()
        }
        self._course_entry_base_totals = {
            course_key: sum(self._entry_base_weights[entry_id] for entry_id in entry_ids)
            for course_key, entry_ids in self._course_entry_ids.items()
        }
        self._course_log_keys = course_log_keys
        self._course_labels = course_labels
        self._stats = {
            course_key: _TrackStepStats(
                base_weight=course_base_weight_sums[course_key]
                / max(course_base_weight_counts[course_key], 1),
                current_weight=course_base_weight_sums[course_key]
                / max(course_base_weight_counts[course_key], 1),
            )
            for course_key in self._course_entry_ids
        }
        if not _uses_dynamic_runtime_mode(sampling_mode):
            raise ValueError(f"Unsupported dynamic track sampling mode: {sampling_mode!r}")
        self._sampling_mode = sampling_mode
        self._action_repeat = max(1, int(action_repeat))
        self._update_episodes = max(1, int(update_episodes))
        self._ema_alpha = max(0.0, min(1.0, float(ema_alpha)))
        self._max_weight_scale = max(1.0, float(max_weight_scale))
        self._adaptive_completion_weight = max(0.0, float(adaptive_completion_weight))
        self._adaptive_target_completion = max(0.0, min(1.0, float(adaptive_target_completion)))
        self._log_details = log_details
        self._episodes_since_update = 0
        self.update_count = 0
        self._restore_state(restored_state)

    @classmethod
    def from_configs(
        cls,
        *,
        env_config: EnvConfig,
        curriculum_config: CurriculumConfig,
        restored_state: TrackSamplingRuntimeState | None = None,
    ) -> StepBalancedTrackSamplingController | None:
        configs = _dynamic_step_balanced_sampling_configs(env_config, curriculum_config)
        if not configs:
            return None

        base_weights: dict[str, float] = {}
        requested_course_keys: dict[str, str] = {}
        requested_log_keys: dict[str, str] = {}
        requested_labels: dict[str, str] = {}
        for config in configs:
            for entry in config.entries:
                base_weights.setdefault(entry.id, float(entry.weight))
                course_key = entry.course_id or entry.id
                requested_course_keys.setdefault(entry.id, course_key)
                requested_log_keys.setdefault(entry.id, course_key)
                requested_labels.setdefault(
                    entry.id,
                    entry.course_name or entry.course_id or entry.display_name or entry.id,
                )
        if len(set(requested_course_keys.values())) <= 1:
            return None

        settings = configs[0]
        return cls(
            track_base_weights=base_weights,
            sampling_mode=settings.sampling_mode,
            action_repeat=env_config.action_repeat,
            update_episodes=settings.step_balance_update_episodes,
            ema_alpha=settings.step_balance_ema_alpha,
            max_weight_scale=settings.step_balance_max_weight_scale,
            adaptive_completion_weight=settings.adaptive_step_balance_completion_weight,
            adaptive_target_completion=settings.adaptive_step_balance_target_completion,
            log_details=settings.step_balance_log_details,
            track_course_keys=requested_course_keys,
            track_log_keys=requested_log_keys,
            track_labels=requested_labels,
            restored_state=restored_state,
        )

    def record_episodes(self, episodes: Sequence[dict[str, object]]) -> dict[str, float] | None:
        recorded_count = 0
        for episode in episodes:
            track_id = _episode_track_id(episode)
            if track_id is None:
                continue
            course_key = self._entry_course_keys.get(track_id, track_id)
            if course_key not in self._stats:
                continue
            frame_count = _episode_frame_count(episode, action_repeat=self._action_repeat)
            if frame_count is None:
                continue
            self._stats[course_key].record_episode(
                frame_count,
                ema_alpha=self._ema_alpha,
                completion_fraction=_episode_completion_fraction(episode),
                finished=_episode_finished(episode),
            )
            recorded_count += 1

        if recorded_count == 0:
            return None

        self._episodes_since_update += recorded_count
        if self._episodes_since_update < self._update_episodes:
            return None

        self._episodes_since_update = 0
        self.update_count += 1
        return self._compute_weights()

    def log_values(self) -> dict[str, float]:
        if not self._log_details:
            return {}

        total_weight = sum(stats.current_weight for stats in self._stats.values())
        values: dict[str, float] = {}
        for course_key, stats in self._stats.items():
            key = self._course_log_keys[course_key]
            values[f"track_sampling/{key}/prob"] = (
                stats.current_weight / total_weight if total_weight > 0.0 else 0.0
            )
        return values

    def current_weights(self) -> dict[str, float]:
        weights: dict[str, float] = {}
        for course_key, stats in self._stats.items():
            weights.update(
                _distribute_course_weight(
                    course_weight=stats.current_weight,
                    entry_ids=self._course_entry_ids[course_key],
                    entry_base_weights=self._entry_base_weights,
                    total_entry_base_weight=self._course_entry_base_totals[course_key],
                )
            )
        return weights

    def runtime_state(self) -> TrackSamplingRuntimeState:
        return TrackSamplingRuntimeState(
            sampling_mode=self._sampling_mode,
            action_repeat=self._action_repeat,
            update_episodes=self._update_episodes,
            ema_alpha=self._ema_alpha,
            max_weight_scale=self._max_weight_scale,
            adaptive_completion_weight=self._adaptive_completion_weight,
            adaptive_target_completion=self._adaptive_target_completion,
            update_count=self.update_count,
            episodes_since_update=self._episodes_since_update,
            entries=tuple(
                TrackSamplingRuntimeEntry(
                    track_id=course_key,
                    course_key=course_key,
                    label=self._course_labels[course_key],
                    base_weight=stats.base_weight,
                    current_weight=stats.current_weight,
                    completed_frames=stats.completed_frames,
                    episode_count=stats.episode_count,
                    finished_episode_count=stats.finished_episode_count,
                    success_sample_count=stats.success_sample_count,
                    ema_episode_frames=stats.ema_episode_frames,
                    ema_completion_fraction=stats.ema_completion_fraction,
                )
                for course_key, stats in sorted(self._stats.items())
            ),
        )

    def _compute_weights(self) -> dict[str, float]:
        reference_length = self._reference_episode_length()
        raw_weights: dict[str, float] = {}
        for track_id, stats in self._stats.items():
            length = stats.ema_episode_frames
            step_scale = 1.0 if length is None else reference_length / max(length, 1.0)
            total_scale = _clamp_weight_scale(step_scale, max_scale=self._max_weight_scale)
            if self._sampling_mode == "adaptive_step_balanced":
                total_scale = _clamp_weight_scale(
                    total_scale * self._adaptive_completion_bonus(stats),
                    max_scale=self._max_weight_scale,
                )
            raw_weights[track_id] = stats.base_weight * total_scale

        total_base_weight = sum(stats.base_weight for stats in self._stats.values())
        total_raw_weight = sum(raw_weights.values())
        if total_raw_weight <= 0.0:
            return self.current_weights()

        normalized = {
            track_id: weight * total_base_weight / total_raw_weight
            for track_id, weight in raw_weights.items()
        }
        for track_id, weight in normalized.items():
            self._stats[track_id].current_weight = weight
        return self.current_weights()

    def _reference_episode_length(self) -> float:
        weighted_lengths = [
            (stats.base_weight, stats.ema_episode_frames)
            for stats in self._stats.values()
            if stats.ema_episode_frames is not None
        ]
        if not weighted_lengths:
            return 1.0
        total_weight = sum(weight for weight, _ in weighted_lengths)
        if total_weight <= 0.0:
            return 1.0
        return max(
            1.0,
            sum(weight * length for weight, length in weighted_lengths) / total_weight,
        )

    def _restore_state(self, restored_state: TrackSamplingRuntimeState | None) -> None:
        if restored_state is None or not _uses_dynamic_runtime_mode(restored_state.sampling_mode):
            return
        state_by_course_key = {
            entry.course_key: entry for entry in _aggregate_runtime_entries(restored_state.entries)
        }
        for course_key, stats in self._stats.items():
            entry = state_by_course_key.get(course_key)
            if entry is None:
                continue
            stats.completed_frames = max(0, int(entry.completed_frames))
            stats.episode_count = max(0, int(entry.episode_count))
            stats.finished_episode_count = max(0, int(entry.finished_episode_count))
            stats.success_sample_count = max(0, int(entry.success_sample_count))
            stats.ema_episode_frames = (
                None
                if entry.ema_episode_frames is None
                else max(0.0, float(entry.ema_episode_frames))
            )
            stats.ema_completion_fraction = (
                None
                if entry.ema_completion_fraction is None
                else max(0.0, min(1.0, float(entry.ema_completion_fraction)))
            )
            stats.current_weight = max(0.0, float(entry.current_weight))
        self._episodes_since_update = max(0, int(restored_state.episodes_since_update))
        self.update_count = max(0, int(restored_state.update_count))

    def _adaptive_completion_bonus(self, stats: _TrackStepStats) -> float:
        return adaptive_difficulty_bonus(
            sampling_mode=self._sampling_mode,
            max_weight_scale=self._max_weight_scale,
            completion_weight=self._adaptive_completion_weight,
            target_completion=self._adaptive_target_completion,
            update_episodes=self._update_episodes,
            completion_fraction=stats.ema_completion_fraction,
            finished_episode_count=stats.finished_episode_count,
            success_sample_count=stats.success_sample_count,
        )


def _distribute_course_weight(
    *,
    course_weight: float,
    entry_ids: Sequence[str],
    entry_base_weights: Mapping[str, float],
    total_entry_base_weight: float,
) -> dict[str, float]:
    if total_entry_base_weight > 0.0:
        return {
            entry_id: course_weight * entry_base_weights[entry_id] / total_entry_base_weight
            for entry_id in entry_ids
        }
    if not entry_ids:
        return {}
    equal_weight = course_weight / len(entry_ids)
    return {entry_id: equal_weight for entry_id in entry_ids}


def adaptive_difficulty_bonus(
    *,
    sampling_mode: str,
    max_weight_scale: float,
    completion_weight: float,
    target_completion: float,
    update_episodes: int,
    completion_fraction: float | None,
    finished_episode_count: int,
    success_sample_count: int,
) -> float:
    if sampling_mode != "adaptive_step_balanced":
        return 1.0
    if max_weight_scale <= 1.0 or completion_weight <= 0.0 or target_completion <= 0.0:
        return 1.0
    completion_gap = _normalized_completion_gap(
        observed_completion=completion_fraction,
        target_completion=target_completion,
    )
    finish_gap = _normalized_completion_gap(
        observed_completion=_observed_finish_rate(
            finished_episode_count=finished_episode_count,
            success_sample_count=success_sample_count,
        ),
        target_completion=target_completion,
    )
    difficulty_signal = max(
        completion_gap,
        _finish_rate_confidence(
            success_sample_count=success_sample_count,
            update_episodes=update_episodes,
        )
        * finish_gap,
    )
    bonus = 1.0 + (max_weight_scale - 1.0) * completion_weight * difficulty_signal
    return min(bonus, max_weight_scale)


def _normalized_completion_gap(
    *,
    observed_completion: float | None,
    target_completion: float,
) -> float:
    if observed_completion is None or target_completion <= 0.0:
        return 0.0
    return max(target_completion - observed_completion, 0.0) / target_completion


def _observed_finish_rate(
    *,
    finished_episode_count: int,
    success_sample_count: int,
) -> float | None:
    if success_sample_count <= 0:
        return None
    return max(0.0, min(1.0, finished_episode_count / success_sample_count))


def _finish_rate_confidence(
    *,
    success_sample_count: int,
    update_episodes: int,
) -> float:
    confidence_episodes = max(1, int(update_episodes)) * 4
    return max(0.0, min(1.0, success_sample_count / confidence_episodes))


def _aggregate_runtime_entries(
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
        left.ema_completion_fraction * left_weight
        + right.ema_completion_fraction * right_weight
    ) / total_weight


def _dynamic_step_balanced_sampling_configs(
    env_config: EnvConfig,
    curriculum_config: CurriculumConfig,
) -> tuple[TrackSamplingConfig, ...]:
    configs: list[TrackSamplingConfig] = []
    if _uses_dynamic_step_balancing(env_config.track_sampling):
        configs.append(env_config.track_sampling)
    if curriculum_config.enabled:
        for stage in curriculum_config.stages:
            if (
                stage.track_sampling is not None
                and _uses_dynamic_step_balancing(stage.track_sampling)
            ):
                configs.append(stage.track_sampling)
    return tuple(configs)


def _uses_dynamic_step_balancing(config: TrackSamplingConfig) -> bool:
    return (
        config.enabled
        and _uses_dynamic_runtime_mode(config.sampling_mode)
        and bool(config.entries)
    )


def _episode_track_id(episode: dict[str, object]) -> str | None:
    value = episode.get("track_id")
    if isinstance(value, str) and value:
        return value
    return None


def _episode_frame_count(
    episode: dict[str, object],
    *,
    action_repeat: int,
) -> int | None:
    episode_step = episode.get("episode_step")
    if isinstance(episode_step, int | float) and not isinstance(episode_step, bool):
        frame_count = int(episode_step)
        return frame_count if frame_count > 0 else None

    monitor_length = episode.get("l")
    if isinstance(monitor_length, int | float) and not isinstance(monitor_length, bool):
        frame_count = int(monitor_length) * action_repeat
        return frame_count if frame_count > 0 else None
    return None


def _episode_finished(episode: Mapping[str, object]) -> bool:
    return episode.get("termination_reason") == "finished"


def _episode_completion_fraction(episode: Mapping[str, object]) -> float | None:
    value = episode.get("episode_completion_fraction")
    if isinstance(value, int | float) and not isinstance(value, bool):
        return max(0.0, min(1.0, float(value)))
    if _episode_finished(episode):
        return 1.0
    return None


def _uses_dynamic_runtime_mode(sampling_mode: str) -> bool:
    return sampling_mode in {"step_balanced", "adaptive_step_balanced"}


def _clamp_weight_scale(scale: float, *, max_scale: float) -> float:
    return max(1.0 / max_scale, min(max_scale, scale))


def _sanitize_log_key(value: str) -> str:
    sanitized = "".join(char if char.isalnum() else "_" for char in value.strip().lower())
    return sanitized.strip("_") or "unknown"


def save_track_sampling_runtime_state(
    path: Path,
    state: TrackSamplingRuntimeState,
) -> None:
    """Persist one dynamic step-balanced sampler snapshot atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.stem}.{os.getpid()}.tmp{path.suffix}")
    data = {
        "version": 1,
        "sampling_mode": state.sampling_mode,
        "action_repeat": state.action_repeat,
        "update_episodes": state.update_episodes,
        "ema_alpha": state.ema_alpha,
        "max_weight_scale": state.max_weight_scale,
        "adaptive_completion_weight": state.adaptive_completion_weight,
        "adaptive_target_completion": state.adaptive_target_completion,
        "update_count": state.update_count,
        "episodes_since_update": state.episodes_since_update,
        "entries": [
            {
                "track_id": entry.track_id,
                "course_key": entry.course_key,
                "label": entry.label,
                "base_weight": entry.base_weight,
                "current_weight": entry.current_weight,
                "completed_frames": entry.completed_frames,
                "episode_count": entry.episode_count,
                "finished_episode_count": entry.finished_episode_count,
                "success_sample_count": entry.success_sample_count,
                "ema_episode_frames": entry.ema_episode_frames,
                "ema_completion_fraction": entry.ema_completion_fraction,
            }
            for entry in state.entries
        ],
    }
    try:
        tmp_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def load_track_sampling_runtime_state(path: Path) -> TrackSamplingRuntimeState | None:
    """Load one persisted dynamic step-balanced sampler snapshot, if present."""

    if not path.is_file():
        return None
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, Mapping):
        return None
    sampling_mode = loaded.get("sampling_mode")
    if not isinstance(sampling_mode, str) or not _uses_dynamic_runtime_mode(sampling_mode):
        return None
    raw_entries = loaded.get("entries")
    if not isinstance(raw_entries, list):
        return None
    entries: list[TrackSamplingRuntimeEntry] = []
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, Mapping):
            continue
        track_id = _mapping_str(raw_entry, "track_id")
        course_key = _mapping_str(raw_entry, "course_key")
        label = _mapping_str(raw_entry, "label")
        base_weight = _mapping_float(raw_entry, "base_weight")
        current_weight = _mapping_float(raw_entry, "current_weight")
        completed_frames = _mapping_int(raw_entry, "completed_frames")
        episode_count = _mapping_int(raw_entry, "episode_count")
        finished_episode_count = _mapping_int(raw_entry, "finished_episode_count")
        success_sample_count = _mapping_int(raw_entry, "success_sample_count")
        ema_episode_frames = _mapping_optional_float(raw_entry, "ema_episode_frames")
        ema_completion_fraction = _mapping_optional_float(raw_entry, "ema_completion_fraction")
        if (
            track_id is None
            or course_key is None
            or label is None
            or base_weight is None
            or current_weight is None
            or completed_frames is None
            or episode_count is None
        ):
            continue
        entries.append(
            TrackSamplingRuntimeEntry(
                track_id=track_id,
                course_key=course_key,
                label=label,
                base_weight=base_weight,
                current_weight=current_weight,
                completed_frames=completed_frames,
                episode_count=episode_count,
                finished_episode_count=(
                    0
                    if finished_episode_count is None
                    else max(0, min(finished_episode_count, episode_count))
                ),
                success_sample_count=(
                    0
                    if success_sample_count is None
                    else max(0, min(success_sample_count, episode_count))
                ),
                ema_episode_frames=ema_episode_frames,
                ema_completion_fraction=(
                    None
                    if ema_completion_fraction is None
                    else max(0.0, min(1.0, ema_completion_fraction))
                ),
            )
        )
    if not entries:
        return None
    action_repeat = _mapping_int(loaded, "action_repeat")
    update_episodes = _mapping_int(loaded, "update_episodes")
    ema_alpha = _mapping_float(loaded, "ema_alpha")
    max_weight_scale = _mapping_float(loaded, "max_weight_scale")
    adaptive_completion_weight = _mapping_optional_float(loaded, "adaptive_completion_weight")
    adaptive_target_completion = _mapping_optional_float(loaded, "adaptive_target_completion")
    update_count = _mapping_int(loaded, "update_count")
    episodes_since_update = _mapping_int(loaded, "episodes_since_update")
    if (
        action_repeat is None
        or update_episodes is None
        or ema_alpha is None
        or max_weight_scale is None
        or update_count is None
        or episodes_since_update is None
    ):
        return None
    return TrackSamplingRuntimeState(
        sampling_mode=sampling_mode,
        action_repeat=action_repeat,
        update_episodes=update_episodes,
        ema_alpha=ema_alpha,
        max_weight_scale=max_weight_scale,
        adaptive_completion_weight=(
            0.35 if adaptive_completion_weight is None else max(0.0, adaptive_completion_weight)
        ),
        adaptive_target_completion=(
            0.9
            if adaptive_target_completion is None
            else max(0.0, min(1.0, adaptive_target_completion))
        ),
        update_count=update_count,
        episodes_since_update=episodes_since_update,
        entries=tuple(entries),
    )


def _mapping_str(mapping: Mapping[str, Any], key: str) -> str | None:
    value = mapping.get(key)
    return value if isinstance(value, str) and value else None


def _mapping_int(mapping: Mapping[str, Any], key: str) -> int | None:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return int(value)


def _mapping_float(mapping: Mapping[str, Any], key: str) -> float | None:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _mapping_optional_float(mapping: Mapping[str, Any], key: str) -> float | None:
    value = mapping.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)
