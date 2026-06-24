# src/rl_fzerox/core/training/session/callbacks/track_sampling/persistence.py
from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rl_fzerox.core.runtime_spec.x_cup_slots import GeneratedXCupSlot
from rl_fzerox.core.training.session.callbacks.track_sampling.alt_baselines import (
    TrackSamplingAltBaseline,
)
from rl_fzerox.core.training.session.callbacks.track_sampling.artifacts import (
    TrackSamplingMaterializedArtifact,
)
from rl_fzerox.core.training.session.callbacks.track_sampling.episodes import (
    uses_track_sampling_runtime_mode,
)
from rl_fzerox.core.training.session.callbacks.track_sampling.state import (
    DeficitBudgetCourseSchedulerState,
    DeficitBudgetSchedulerState,
    TrackSamplingRuntimeEntry,
    TrackSamplingRuntimeState,
)


@dataclass(frozen=True, slots=True)
class TrackSamplingRuntimeLoadDefaults:
    """Defaults used when loading older runtime-state snapshots."""

    adaptive_completion_weight: float = 0.35
    adaptive_target_completion: float = 0.9
    adaptive_min_confidence_episodes: int = 24
    adaptive_confidence_scale: float = 4.0
    deficit_budget_difficulty_metric: str = "completion_ema"
    deficit_budget_warmup_min_episodes_per_course: int = 0


_RUNTIME_LOAD_DEFAULTS = TrackSamplingRuntimeLoadDefaults()


@dataclass(frozen=True, slots=True)
class TrackSamplingRuntimePersistence:
    """Load/save boundary for manager DB or standalone file-backed state."""

    load: Callable[[], TrackSamplingRuntimeState | None]
    save: Callable[[TrackSamplingRuntimeState], None]
    replace_materialized_artifacts: (
        Callable[[tuple[TrackSamplingMaterializedArtifact, ...]], None] | None
    ) = None
    replace_generated_x_cup_slots: Callable[[tuple[GeneratedXCupSlot, ...]], None] | None = None
    load_alt_baselines: Callable[[], tuple[TrackSamplingAltBaseline, ...]] | None = None


def file_track_sampling_runtime_persistence(path: Path) -> TrackSamplingRuntimePersistence:
    """Return standalone file-backed persistence for non-manager runs."""

    return TrackSamplingRuntimePersistence(
        load=lambda: load_track_sampling_runtime_state(path),
        save=lambda state: save_track_sampling_runtime_state(path, state),
    )


def save_track_sampling_runtime_state(
    path: Path,
    state: TrackSamplingRuntimeState,
) -> None:
    """Persist one dynamic step-balanced sampler snapshot atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.stem}.{os.getpid()}.tmp{path.suffix}")
    try:
        tmp_path.write_text(
            track_sampling_runtime_state_json(state),
            encoding="utf-8",
        )
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def track_sampling_runtime_state_json(state: TrackSamplingRuntimeState) -> str:
    """Serialize one sampler state snapshot for DB or file persistence.

    Keep entry fields in lockstep with TrackSamplingRuntimeEntry. Missing
    optional fields are handled only in the JSON loader so old snapshots have an
    explicit compatibility boundary.
    """

    data = {
        "version": 1,
        "sampling_mode": state.sampling_mode,
        "action_repeat": state.action_repeat,
        "update_episodes": state.update_episodes,
        "ema_alpha": state.ema_alpha,
        "max_weight_scale": state.max_weight_scale,
        "adaptive_completion_weight": state.adaptive_completion_weight,
        "adaptive_target_completion": state.adaptive_target_completion,
        "adaptive_min_confidence_episodes": state.adaptive_min_confidence_episodes,
        "adaptive_confidence_scale": state.adaptive_confidence_scale,
        "deficit_budget_difficulty_metric": state.deficit_budget_difficulty_metric,
        "deficit_budget_warmup_min_episodes_per_course": (
            state.deficit_budget_warmup_min_episodes_per_course
        ),
        "deficit_budget_scheduler": _deficit_budget_scheduler_data(
            state.deficit_budget_scheduler,
        ),
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
                "completion_sample_count": entry.completion_sample_count,
                "completion_fraction_total": entry.completion_fraction_total,
                "ema_episode_frames": entry.ema_episode_frames,
                "ema_completion_fraction": entry.ema_completion_fraction,
                "ema_finish_rate": entry.ema_finish_rate,
                "current_problem_score": entry.current_problem_score,
                "generation_episode_count": entry.generation_episode_count,
                "generation_finished_episode_count": entry.generation_finished_episode_count,
                "generation_success_sample_count": entry.generation_success_sample_count,
                "generation_ema_completion_fraction": entry.generation_ema_completion_fraction,
                "generated_course_slot": entry.generated_course_slot,
                "generated_course_generation": entry.generated_course_generation,
                "generated_course_id": entry.generated_course_id,
                "generated_course_name": entry.generated_course_name,
                "generated_course_hash": entry.generated_course_hash,
                "generated_course_seed": entry.generated_course_seed,
                "generated_course_segment_count": entry.generated_course_segment_count,
                "generated_course_length": entry.generated_course_length,
            }
            for entry in state.entries
        ],
    }
    return json.dumps(data, indent=2, sort_keys=True) + "\n"


def load_track_sampling_runtime_state(path: Path) -> TrackSamplingRuntimeState | None:
    """Load one persisted dynamic step-balanced sampler snapshot, if present."""

    if not path.is_file():
        return None
    return load_track_sampling_runtime_state_json(path.read_text(encoding="utf-8"))


def load_track_sampling_runtime_state_json(data: str) -> TrackSamplingRuntimeState | None:
    """Load one persisted dynamic step-balanced sampler snapshot from JSON text."""

    loaded = json.loads(data)
    if not isinstance(loaded, Mapping):
        return None
    sampling_mode = loaded.get("sampling_mode")
    if not isinstance(sampling_mode, str) or not uses_track_sampling_runtime_mode(sampling_mode):
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
    adaptive_min_confidence_episodes = _mapping_int(loaded, "adaptive_min_confidence_episodes")
    adaptive_confidence_scale = _mapping_optional_float(loaded, "adaptive_confidence_scale")
    deficit_budget_difficulty_metric = _mapping_optional_str(
        loaded,
        "deficit_budget_difficulty_metric",
    )
    deficit_budget_warmup_min_episodes_per_course = _mapping_int(
        loaded,
        "deficit_budget_warmup_min_episodes_per_course",
    )
    deficit_budget_scheduler = _deficit_budget_scheduler_from_data(
        loaded.get("deficit_budget_scheduler"),
    )
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
        adaptive_completion_weight=_min_float_or_default(
            adaptive_completion_weight,
            default=_RUNTIME_LOAD_DEFAULTS.adaptive_completion_weight,
            minimum=0.0,
        ),
        adaptive_target_completion=_clamped_float_or_default(
            adaptive_target_completion,
            default=_RUNTIME_LOAD_DEFAULTS.adaptive_target_completion,
            minimum=0.0,
            maximum=1.0,
        ),
        adaptive_min_confidence_episodes=_min_int_or_default(
            adaptive_min_confidence_episodes,
            default=_RUNTIME_LOAD_DEFAULTS.adaptive_min_confidence_episodes,
            minimum=1,
        ),
        adaptive_confidence_scale=_min_float_or_default(
            adaptive_confidence_scale,
            default=_RUNTIME_LOAD_DEFAULTS.adaptive_confidence_scale,
            minimum=1.0,
        ),
        update_count=update_count,
        episodes_since_update=episodes_since_update,
        entries=tuple(entries),
        deficit_budget_difficulty_metric=_deficit_budget_difficulty_metric(
            deficit_budget_difficulty_metric,
        ),
        deficit_budget_warmup_min_episodes_per_course=_min_int_or_default(
            deficit_budget_warmup_min_episodes_per_course,
            default=_RUNTIME_LOAD_DEFAULTS.deficit_budget_warmup_min_episodes_per_course,
            minimum=0,
        ),
        deficit_budget_scheduler=deficit_budget_scheduler,
    )


def _runtime_entries_from_data(raw_entries: list[object]) -> list[TrackSamplingRuntimeEntry]:
    """Decode persisted entry JSON and clamp counters back to model invariants."""

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
        raw_completion_sample_count = _mapping_int(raw_entry, "completion_sample_count")
        completion_fraction_total = _mapping_optional_float(
            raw_entry,
            "completion_fraction_total",
        )
        ema_episode_frames = _mapping_optional_float(raw_entry, "ema_episode_frames")
        ema_completion_fraction = _mapping_optional_float(raw_entry, "ema_completion_fraction")
        ema_finish_rate = _mapping_optional_float(raw_entry, "ema_finish_rate")
        current_problem_score = _mapping_optional_float(raw_entry, "current_problem_score")
        generation_episode_count = _mapping_int(raw_entry, "generation_episode_count")
        generation_finished_episode_count = _mapping_int(
            raw_entry,
            "generation_finished_episode_count",
        )
        generation_success_sample_count = _mapping_int(
            raw_entry,
            "generation_success_sample_count",
        )
        generation_ema_completion_fraction = _mapping_optional_float(
            raw_entry,
            "generation_ema_completion_fraction",
        )
        generated_course_slot = _mapping_optional_int(raw_entry, "generated_course_slot")
        generated_course_generation = _mapping_optional_int(
            raw_entry,
            "generated_course_generation",
        )
        generated_course_id = _mapping_optional_str(raw_entry, "generated_course_id")
        generated_course_name = _mapping_optional_str(raw_entry, "generated_course_name")
        generated_course_hash = _mapping_optional_str(raw_entry, "generated_course_hash")
        generated_course_seed = _mapping_optional_int(raw_entry, "generated_course_seed")
        generated_course_segment_count = _mapping_optional_int(
            raw_entry,
            "generated_course_segment_count",
        )
        generated_course_length = _mapping_optional_float(
            raw_entry,
            "generated_course_length",
        )
        normalized_completion_sample_count = (
            0 if success_sample_count is None else max(0, success_sample_count)
        )
        completion_sample_count = (
            normalized_completion_sample_count
            if raw_completion_sample_count is None
            else raw_completion_sample_count
        )
        if completion_fraction_total is None:
            completion_fraction_total = (
                0.0
                if ema_completion_fraction is None
                else max(0.0, min(1.0, ema_completion_fraction)) * max(0, completion_sample_count)
            )
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
        clamped_completion_sample_count = max(0, min(completion_sample_count, episode_count))
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
                completion_sample_count=clamped_completion_sample_count,
                completion_fraction_total=max(
                    0.0,
                    min(float(clamped_completion_sample_count), completion_fraction_total),
                ),
                ema_episode_frames=ema_episode_frames,
                ema_completion_fraction=(
                    None
                    if ema_completion_fraction is None
                    else max(0.0, min(1.0, ema_completion_fraction))
                ),
                ema_finish_rate=(
                    None if ema_finish_rate is None else max(0.0, min(1.0, ema_finish_rate))
                ),
                current_problem_score=(
                    0.0
                    if current_problem_score is None
                    else max(0.0, min(1.0, current_problem_score))
                ),
                generation_episode_count=(
                    0 if generation_episode_count is None else max(0, generation_episode_count)
                ),
                generation_finished_episode_count=(
                    0
                    if generation_finished_episode_count is None
                    else max(
                        0,
                        min(
                            generation_finished_episode_count,
                            0 if generation_episode_count is None else generation_episode_count,
                        ),
                    )
                ),
                generation_success_sample_count=(
                    0
                    if generation_success_sample_count is None
                    else max(
                        0,
                        min(
                            generation_success_sample_count,
                            0 if generation_episode_count is None else generation_episode_count,
                        ),
                    )
                ),
                generation_ema_completion_fraction=(
                    None
                    if generation_ema_completion_fraction is None
                    else max(0.0, min(1.0, generation_ema_completion_fraction))
                ),
                generated_course_slot=(
                    None if generated_course_slot is None else max(0, generated_course_slot)
                ),
                generated_course_generation=(
                    None
                    if generated_course_generation is None
                    else max(1, generated_course_generation)
                ),
                generated_course_id=generated_course_id,
                generated_course_name=generated_course_name,
                generated_course_hash=generated_course_hash,
                generated_course_seed=(
                    None if generated_course_seed is None else max(0, generated_course_seed)
                ),
                generated_course_segment_count=(
                    None
                    if generated_course_segment_count is None
                    else max(0, generated_course_segment_count)
                ),
                generated_course_length=(
                    None if generated_course_length is None else max(0.0, generated_course_length)
                ),
            )
        )
    return entries


def deficit_budget_scheduler_state_json(
    state: DeficitBudgetSchedulerState | None,
) -> str | None:
    """Serialize deficit-budget scheduler internals for manager DB storage."""

    data = _deficit_budget_scheduler_data(state)
    return None if data is None else json.dumps(data, sort_keys=True)


def load_deficit_budget_scheduler_state_json(
    data: str | None,
) -> DeficitBudgetSchedulerState | None:
    """Load deficit-budget scheduler internals from manager DB storage."""

    if not data:
        return None
    loaded = json.loads(data)
    return _deficit_budget_scheduler_from_data(loaded)


def _deficit_budget_scheduler_data(
    state: DeficitBudgetSchedulerState | None,
) -> dict[str, object] | None:
    if state is None:
        return None
    return {
        "uniform_lane_deficit_steps": state.uniform_lane_deficit_steps,
        "adaptive_lane_deficit_steps": state.adaptive_lane_deficit_steps,
        "uniform_assignment_count": state.uniform_assignment_count,
        "entries": [
            {
                "course_key": entry.course_key,
                "uniform_deficit_steps": entry.uniform_deficit_steps,
                "adaptive_deficit_steps": entry.adaptive_deficit_steps,
                "scheduler_env_steps": entry.scheduler_env_steps,
                "last_uniform_assignment_index": entry.last_uniform_assignment_index,
            }
            for entry in state.entries
        ],
    }


def _deficit_budget_scheduler_from_data(
    data: object,
) -> DeficitBudgetSchedulerState | None:
    if not isinstance(data, Mapping):
        return None
    raw_entries = data.get("entries")
    if not isinstance(raw_entries, list):
        return None
    entries: list[DeficitBudgetCourseSchedulerState] = []
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, Mapping):
            continue
        course_key = _mapping_str(raw_entry, "course_key")
        if course_key is None:
            continue
        entries.append(
            DeficitBudgetCourseSchedulerState(
                course_key=course_key,
                uniform_deficit_steps=_mapping_optional_float(
                    raw_entry,
                    "uniform_deficit_steps",
                )
                or 0.0,
                adaptive_deficit_steps=_mapping_optional_float(
                    raw_entry,
                    "adaptive_deficit_steps",
                )
                or 0.0,
                scheduler_env_steps=max(
                    0,
                    _mapping_optional_int(raw_entry, "scheduler_env_steps") or 0,
                ),
                last_uniform_assignment_index=max(
                    0,
                    _mapping_optional_int(raw_entry, "last_uniform_assignment_index") or 0,
                ),
            )
        )
    if not entries:
        return None
    return DeficitBudgetSchedulerState(
        uniform_lane_deficit_steps=_mapping_optional_float(
            data,
            "uniform_lane_deficit_steps",
        )
        or 0.0,
        adaptive_lane_deficit_steps=_mapping_optional_float(
            data,
            "adaptive_lane_deficit_steps",
        )
        or 0.0,
        uniform_assignment_count=max(
            0,
            _mapping_optional_int(data, "uniform_assignment_count") or 0,
        ),
        entries=tuple(sorted(entries, key=lambda entry: entry.course_key)),
    )


def _mapping_str(mapping: Mapping[str, Any], key: str) -> str | None:
    value = mapping.get(key)
    return value if isinstance(value, str) and value else None


def _mapping_int(mapping: Mapping[str, Any], key: str) -> int | None:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return int(value)


def _mapping_optional_int(mapping: Mapping[str, Any], key: str) -> int | None:
    value = mapping.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return int(value)


def _mapping_optional_str(mapping: Mapping[str, Any], key: str) -> str | None:
    value = mapping.get(key)
    return value if isinstance(value, str) and value else None


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


def _min_int_or_default(
    value: int | None,
    *,
    default: int,
    minimum: int,
) -> int:
    return default if value is None else max(minimum, value)


def _min_float_or_default(
    value: float | None,
    *,
    default: float,
    minimum: float,
) -> float:
    return default if value is None else max(minimum, value)


def _clamped_float_or_default(
    value: float | None,
    *,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    return default if value is None else max(minimum, min(maximum, value))


def _deficit_budget_difficulty_metric(value: str | None) -> str:
    if value in {"completion_ema", "finish_ema", "mixed"}:
        return value
    return _RUNTIME_LOAD_DEFAULTS.deficit_budget_difficulty_metric
