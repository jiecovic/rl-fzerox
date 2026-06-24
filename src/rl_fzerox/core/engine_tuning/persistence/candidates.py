# src/rl_fzerox/core/engine_tuning/persistence/candidates.py
"""Candidate-state payload conversion for engine-tuning checkpoints."""

from __future__ import annotations

from collections.abc import Mapping

from rl_fzerox.core.engine_tuning.persistence.fields import (
    mapping_float,
    mapping_int,
    mapping_optional_float,
    mapping_optional_int,
    mapping_str,
)
from rl_fzerox.core.engine_tuning.state import EngineTuningCandidateState


def candidate_payload(candidate: EngineTuningCandidateState) -> dict[str, object]:
    """Return the JSON-safe payload for one candidate aggregate."""

    return {
        "context_key": candidate.context_key,
        "course_key": candidate.course_key,
        "vehicle_id": candidate.vehicle_id,
        "engine_setting_raw_value": candidate.engine_setting_raw_value,
        "score_count": candidate.score_count,
        "episode_count": candidate.episode_count,
        "finish_count": candidate.finish_count,
        "return_count": candidate.return_count,
        "decayed_count": candidate.decayed_count,
        "decayed_score_total": candidate.decayed_score_total,
        "score_total": candidate.score_total,
        "best_score": candidate.best_score,
        "completion_score_total": candidate.completion_score_total,
        "best_completion_score": candidate.best_completion_score,
        "finish_score_total": candidate.finish_score_total,
        "best_finish_score": candidate.best_finish_score,
        "return_score_total": candidate.return_score_total,
        "best_return_score": candidate.best_return_score,
        "best_time_ms": candidate.best_time_ms,
    }


def candidate_from_mapping(raw: object) -> EngineTuningCandidateState | None:
    """Decode one candidate aggregate from a JSON-like mapping."""

    if not isinstance(raw, Mapping):
        return None
    context_key = mapping_str(raw, "context_key")
    course_key = mapping_str(raw, "course_key")
    vehicle_id = mapping_str(raw, "vehicle_id")
    engine_setting_raw_value = mapping_int(raw, "engine_setting_raw_value")
    if (
        context_key is None
        or course_key is None
        or vehicle_id is None
        or engine_setting_raw_value is None
    ):
        return None
    finish_count = max(0, mapping_int(raw, "finish_count") or 0)
    score_count = mapping_int(raw, "score_count")
    raw_episode_count = mapping_int(raw, "episode_count")
    episode_count = None if raw_episode_count is None else max(0, raw_episode_count)
    return_count = mapping_int(raw, "return_count")
    score_total = mapping_float(raw, "score_total") or 0.0
    best_score = mapping_optional_float(raw, "best_score")
    finish_score_total = mapping_float(raw, "finish_score_total")
    best_finish_score = mapping_optional_float(raw, "best_finish_score")
    completion_score_total = mapping_float(raw, "completion_score_total")
    best_completion_score = mapping_optional_float(raw, "best_completion_score")
    return_score_total = mapping_float(raw, "return_score_total") or 0.0
    best_return_score = mapping_optional_float(raw, "best_return_score")
    inferred_return_count = 0 if return_count is None else max(0, return_count)
    inferred_episode_count = max(
        0, inferred_return_count if episode_count is None else episode_count
    )
    inferred_completion_total = (
        completion_score_total if completion_score_total is not None else 0.0
    )
    resolved_finish_score_total = finish_score_total
    if resolved_finish_score_total is None:
        resolved_finish_score_total = 0.0 if "finish_score_total" in raw else score_total
    return EngineTuningCandidateState(
        context_key=context_key,
        course_key=course_key,
        vehicle_id=vehicle_id,
        engine_setting_raw_value=engine_setting_raw_value,
        score_count=max(0, score_count if score_count is not None else finish_count),
        episode_count=inferred_episode_count,
        finish_count=finish_count,
        return_count=inferred_return_count,
        decayed_count=max(0.0, mapping_float(raw, "decayed_count") or 0.0),
        decayed_score_total=mapping_float(raw, "decayed_score_total") or 0.0,
        score_total=score_total,
        best_score=best_score,
        completion_score_total=max(0.0, inferred_completion_total),
        best_completion_score=_clamp_unit_score(best_completion_score),
        finish_score_total=resolved_finish_score_total,
        best_finish_score=(
            best_finish_score
            if best_finish_score is not None or "best_finish_score" in raw
            else best_score
        ),
        return_score_total=return_score_total,
        best_return_score=best_return_score,
        best_time_ms=mapping_optional_int(raw, "best_time_ms"),
    )


def _clamp_unit_score(value: float | None) -> float | None:
    if value is None:
        return None
    return max(0.0, min(1.0, float(value)))
