# src/rl_fzerox/core/training/session/callbacks/track_sampling/persistence.py
from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from rl_fzerox.core.training.session.callbacks.track_sampling.episodes import (
    uses_dynamic_runtime_mode,
)
from rl_fzerox.core.training.session.callbacks.track_sampling.state import (
    TrackSamplingRuntimeEntry,
    TrackSamplingRuntimeState,
)


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
    if not isinstance(sampling_mode, str) or not uses_dynamic_runtime_mode(sampling_mode):
        return None
    raw_entries = loaded.get("entries")
    if not isinstance(raw_entries, list):
        return None
    entries = _runtime_entries_from_data(raw_entries)
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


def _runtime_entries_from_data(raw_entries: list[object]) -> list[TrackSamplingRuntimeEntry]:
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
    return entries


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
