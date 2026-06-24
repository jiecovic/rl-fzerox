# src/rl_fzerox/core/engine_tuning/contexts.py
from __future__ import annotations

from rl_fzerox.core.domain.x_cup import X_CUP_COURSE
from rl_fzerox.core.engine_tuning.types import EngineTuningContext
from rl_fzerox.core.runtime_spec.schema import TrackSamplingConfig, TrackSamplingEntryConfig


def engine_tuning_context_for_entry(entry: TrackSamplingEntryConfig) -> EngineTuningContext:
    """Return the adaptive engine-tuning context for one materialized reset entry."""

    return EngineTuningContext(
        course_key=_engine_tuning_course_key(entry),
        vehicle_id=entry.vehicle or entry.source_vehicle or "unknown",
    )


def engine_tuning_contexts_for_track_sampling(
    track_sampling: TrackSamplingConfig,
) -> tuple[EngineTuningContext, ...]:
    contexts: dict[str, EngineTuningContext] = {}
    for entry in track_sampling.entries:
        context = engine_tuning_context_for_entry(entry)
        contexts.setdefault(context.key, context)
    return tuple(contexts[key] for key in sorted(contexts))


def _engine_tuning_course_key(entry: TrackSamplingEntryConfig) -> str:
    if entry.generated_course_kind == X_CUP_COURSE.generated_kind:
        return "x_cup"
    return (
        entry.runtime_course_key
        or entry.course_id
        or entry.course_ref
        or (f"course_index:{entry.course_index}" if entry.course_index is not None else entry.id)
    )
