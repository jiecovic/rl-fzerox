# src/rl_fzerox/core/training/session/callbacks/track_sampling.py
from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rl_fzerox.core.config.schema import CurriculumConfig, EnvConfig, TrackSamplingConfig


@dataclass(frozen=True, slots=True)
class TrackSamplingRuntimeEntry:
    track_id: str
    course_key: str
    label: str
    base_weight: float
    current_weight: float
    completed_frames: int
    episode_count: int
    ema_episode_frames: float | None


@dataclass(frozen=True, slots=True)
class TrackSamplingRuntimeState:
    sampling_mode: str
    action_repeat: int
    update_episodes: int
    ema_alpha: float
    max_weight_scale: float
    update_count: int
    episodes_since_update: int
    entries: tuple[TrackSamplingRuntimeEntry, ...]


@dataclass(slots=True)
class _TrackStepStats:
    base_weight: float
    completed_frames: int = 0
    episode_count: int = 0
    ema_episode_frames: float | None = None
    current_weight: float = 1.0

    def record_episode(self, frame_count: int, *, ema_alpha: float) -> None:
        self.completed_frames += frame_count
        self.episode_count += 1
        if self.ema_episode_frames is None:
            self.ema_episode_frames = float(frame_count)
            return
        self.ema_episode_frames = (
            (1.0 - ema_alpha) * self.ema_episode_frames + ema_alpha * frame_count
        )


class StepBalancedTrackSamplingController:
    """Adapt track reset weights so each map trends toward equal frame coverage."""

    def __init__(
        self,
        *,
        track_base_weights: dict[str, float],
        action_repeat: int,
        update_episodes: int,
        ema_alpha: float,
        max_weight_scale: float,
        log_details: bool = False,
        track_log_keys: dict[str, str] | None = None,
        track_labels: dict[str, str] | None = None,
        restored_state: TrackSamplingRuntimeState | None = None,
    ) -> None:
        self._stats = {
            track_id: _TrackStepStats(base_weight=weight, current_weight=weight)
            for track_id, weight in track_base_weights.items()
        }
        self._action_repeat = max(1, int(action_repeat))
        self._update_episodes = max(1, int(update_episodes))
        self._ema_alpha = max(0.0, min(1.0, float(ema_alpha)))
        self._max_weight_scale = max(1.0, float(max_weight_scale))
        self._log_details = log_details
        self._track_log_keys = {
            track_id: _sanitize_log_key((track_log_keys or {}).get(track_id, track_id))
            for track_id in self._stats
        }
        self._track_labels = {
            track_id: (track_labels or {}).get(track_id, track_id)
            for track_id in self._stats
        }
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
        configs = _step_balanced_sampling_configs(env_config, curriculum_config)
        if not configs:
            return None

        base_weights: dict[str, float] = {}
        requested_log_keys: dict[str, str] = {}
        requested_labels: dict[str, str] = {}
        for config in configs:
            for entry in config.entries:
                base_weights.setdefault(entry.id, float(entry.weight))
                requested_log_keys.setdefault(entry.id, entry.course_id or entry.id)
                requested_labels.setdefault(
                    entry.id,
                    entry.course_name or entry.display_name or entry.course_id or entry.id,
                )
        if len(base_weights) <= 1:
            return None

        settings = configs[0]
        return cls(
            track_base_weights=base_weights,
            action_repeat=env_config.action_repeat,
            update_episodes=settings.step_balance_update_episodes,
            ema_alpha=settings.step_balance_ema_alpha,
            max_weight_scale=settings.step_balance_max_weight_scale,
            log_details=settings.step_balance_log_details,
            track_log_keys=requested_log_keys,
            track_labels=requested_labels,
            restored_state=restored_state,
        )

    def record_episodes(self, episodes: Sequence[dict[str, object]]) -> dict[str, float] | None:
        recorded_count = 0
        for episode in episodes:
            track_id = _episode_track_id(episode)
            if track_id is None or track_id not in self._stats:
                continue
            frame_count = _episode_frame_count(episode, action_repeat=self._action_repeat)
            if frame_count is None:
                continue
            self._stats[track_id].record_episode(frame_count, ema_alpha=self._ema_alpha)
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
        course_weights: dict[str, float] = {}

        for track_id, stats in self._stats.items():
            key = self._track_log_keys[track_id]
            course_weights[key] = course_weights.get(key, 0.0) + stats.current_weight
        for key, weight in course_weights.items():
            values[f"track_sampling/{key}/prob"] = (
                weight / total_weight if total_weight > 0.0 else 0.0
            )
        return values

    def current_weights(self) -> dict[str, float]:
        return {
            track_id: stats.current_weight
            for track_id, stats in self._stats.items()
        }

    def runtime_state(self) -> TrackSamplingRuntimeState:
        return TrackSamplingRuntimeState(
            sampling_mode="step_balanced",
            action_repeat=self._action_repeat,
            update_episodes=self._update_episodes,
            ema_alpha=self._ema_alpha,
            max_weight_scale=self._max_weight_scale,
            update_count=self.update_count,
            episodes_since_update=self._episodes_since_update,
            entries=tuple(
                TrackSamplingRuntimeEntry(
                    track_id=track_id,
                    course_key=self._track_log_keys[track_id],
                    label=self._track_labels[track_id],
                    base_weight=stats.base_weight,
                    current_weight=stats.current_weight,
                    completed_frames=stats.completed_frames,
                    episode_count=stats.episode_count,
                    ema_episode_frames=stats.ema_episode_frames,
                )
                for track_id, stats in sorted(self._stats.items())
            ),
        )

    def _compute_weights(self) -> dict[str, float]:
        reference_length = self._reference_episode_length()
        raw_weights: dict[str, float] = {}
        for track_id, stats in self._stats.items():
            length = stats.ema_episode_frames
            scale = 1.0 if length is None else reference_length / max(length, 1.0)
            scale = max(1.0 / self._max_weight_scale, min(self._max_weight_scale, scale))
            raw_weights[track_id] = stats.base_weight * scale

        total_base_weight = sum(stats.base_weight for stats in self._stats.values())
        total_raw_weight = sum(raw_weights.values())
        if total_raw_weight <= 0.0:
            return {
                track_id: stats.current_weight for track_id, stats in self._stats.items()
            }

        normalized = {
            track_id: weight * total_base_weight / total_raw_weight
            for track_id, weight in raw_weights.items()
        }
        for track_id, weight in normalized.items():
            self._stats[track_id].current_weight = weight
        return normalized

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
        if restored_state is None or restored_state.sampling_mode != "step_balanced":
            return
        state_by_track_id = {entry.track_id: entry for entry in restored_state.entries}
        for track_id, stats in self._stats.items():
            entry = state_by_track_id.get(track_id)
            if entry is None:
                continue
            stats.completed_frames = max(0, int(entry.completed_frames))
            stats.episode_count = max(0, int(entry.episode_count))
            stats.ema_episode_frames = (
                None
                if entry.ema_episode_frames is None
                else max(0.0, float(entry.ema_episode_frames))
            )
            stats.current_weight = max(0.0, float(entry.current_weight))
        self._episodes_since_update = max(0, int(restored_state.episodes_since_update))
        self.update_count = max(0, int(restored_state.update_count))


def _step_balanced_sampling_configs(
    env_config: EnvConfig,
    curriculum_config: CurriculumConfig,
) -> tuple[TrackSamplingConfig, ...]:
    configs: list[TrackSamplingConfig] = []
    if _is_step_balanced(env_config.track_sampling):
        configs.append(env_config.track_sampling)
    if curriculum_config.enabled:
        for stage in curriculum_config.stages:
            if stage.track_sampling is not None and _is_step_balanced(stage.track_sampling):
                configs.append(stage.track_sampling)
    return tuple(configs)


def _is_step_balanced(config: TrackSamplingConfig) -> bool:
    return config.enabled and config.sampling_mode == "step_balanced" and bool(config.entries)


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


def _sanitize_log_key(value: str) -> str:
    sanitized = "".join(char if char.isalnum() else "_" for char in value.strip().lower())
    return sanitized.strip("_") or "unknown"


def save_track_sampling_runtime_state(
    path: Path,
    state: TrackSamplingRuntimeState,
) -> None:
    """Persist one step-balanced sampler snapshot atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.stem}.{os.getpid()}.tmp{path.suffix}")
    data = {
        "version": 1,
        "sampling_mode": state.sampling_mode,
        "action_repeat": state.action_repeat,
        "update_episodes": state.update_episodes,
        "ema_alpha": state.ema_alpha,
        "max_weight_scale": state.max_weight_scale,
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
                "ema_episode_frames": entry.ema_episode_frames,
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
    """Load one persisted step-balanced sampler snapshot, if present."""

    if not path.is_file():
        return None
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, Mapping):
        return None
    if loaded.get("sampling_mode") != "step_balanced":
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
        ema_episode_frames = _mapping_optional_float(raw_entry, "ema_episode_frames")
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
                ema_episode_frames=ema_episode_frames,
            )
        )
    if not entries:
        return None
    action_repeat = _mapping_int(loaded, "action_repeat")
    update_episodes = _mapping_int(loaded, "update_episodes")
    ema_alpha = _mapping_float(loaded, "ema_alpha")
    max_weight_scale = _mapping_float(loaded, "max_weight_scale")
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
        sampling_mode="step_balanced",
        action_repeat=action_repeat,
        update_episodes=update_episodes,
        ema_alpha=ema_alpha,
        max_weight_scale=max_weight_scale,
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
