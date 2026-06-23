# src/rl_fzerox/apps/run_manager/api/payloads/evaluations.py
from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from rl_fzerox.core.evaluation.models import EvaluationCheckpointSnapshot
from rl_fzerox.core.manager import (
    ManagedEvaluation,
    ManagedEvaluationBaselineSuite,
    ManagedEvaluationPreset,
)


def evaluation_payload(
    evaluation: ManagedEvaluation,
    *,
    baseline_suite: ManagedEvaluationBaselineSuite,
) -> dict[str, object]:
    """Return one evaluation record payload."""

    result_file = _read_result_file(evaluation)
    return {
        "id": evaluation.id,
        "name": evaluation.name,
        "status": evaluation.status,
        "evaluation_dir": str(evaluation.evaluation_dir),
        "source_run_id": evaluation.source_run_id,
        "source_artifact": evaluation.source_artifact,
        "preset_id": evaluation.preset_id,
        "preset_version": evaluation.preset_version,
        "policy_mode": evaluation.policy_mode,
        "device": evaluation.config.train.device,
        "seed": evaluation.seed,
        "target": asdict(evaluation.target),
        "config": evaluation.config.model_dump(mode="json"),
        "checkpoint": _checkpoint_payload(evaluation.checkpoint),
        "created_at": evaluation.created_at,
        "updated_at": evaluation.updated_at,
        "started_at": evaluation.started_at,
        "finished_at": evaluation.finished_at,
        "result_json_path": (
            None if evaluation.result_json_path is None else str(evaluation.result_json_path)
        ),
        "error_message": evaluation.error_message,
        "progress": _progress_payload(result_file),
        "result_summary": _result_summary_payload(result_file),
        "baseline_suite": evaluation_baseline_suite_payload(baseline_suite),
    }


def evaluation_preset_payload(preset: ManagedEvaluationPreset) -> dict[str, object]:
    """Return one persisted evaluation-preset payload."""

    return {
        "id": preset.id,
        "name": preset.name,
        "version": preset.version,
        "seed": preset.seed,
        "renderer": preset.renderer,
        "target": asdict(preset.target),
        "builtin": preset.builtin,
        "created_at": preset.created_at,
        "updated_at": preset.updated_at,
    }


def evaluation_baseline_suite_payload(
    suite: ManagedEvaluationBaselineSuite,
) -> dict[str, object]:
    """Return one preset-version baseline-suite payload."""

    return {
        "id": suite.id,
        "preset_id": suite.preset_id,
        "preset_version": suite.preset_version,
        "status": suite.status,
        "suite_dir": str(suite.suite_dir),
        "manifest_path": None if suite.manifest_path is None else str(suite.manifest_path),
        "error_message": suite.error_message,
        "created_at": suite.created_at,
        "updated_at": suite.updated_at,
        "materialized_at": suite.materialized_at,
    }


def _checkpoint_payload(checkpoint: EvaluationCheckpointSnapshot) -> dict[str, object]:
    payload = asdict(checkpoint)
    # Nanosecond mtimes exceed JavaScript's safe integer range. Keep the Python
    # model and persisted spec numeric, but expose the API value losslessly.
    payload["source_mtime_ns"] = (
        None if checkpoint.source_mtime_ns is None else str(checkpoint.source_mtime_ns)
    )
    return payload


def _read_result_file(evaluation: ManagedEvaluation) -> dict[str, Any] | None:
    if evaluation.result_json_path is None or not evaluation.result_json_path.is_file():
        return None
    try:
        payload = json.loads(evaluation.result_json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _progress_payload(payload: dict[str, Any] | None) -> dict[str, object]:
    completed_attempts = 0
    total_attempts: int | None = None
    result_status: str | None = None
    result = payload.get("result") if payload is not None else None
    if isinstance(result, dict):
        attempts = result.get("attempts")
        if isinstance(attempts, list):
            completed_attempts = len(attempts)
        spec = result.get("spec")
        if isinstance(spec, dict):
            planned = spec.get("total_planned_attempts")
            if isinstance(planned, int) and planned >= 0:
                total_attempts = planned
        raw_status = result.get("status")
        if isinstance(raw_status, str):
            result_status = raw_status
    return {
        "completed_attempts": completed_attempts,
        "total_attempts": total_attempts,
        "result_status": result_status,
    }


def _result_summary_payload(payload: dict[str, Any] | None) -> dict[str, object] | None:
    result = payload.get("result") if payload is not None else None
    metrics = payload.get("metrics") if payload is not None else None
    if not isinstance(result, dict) or not isinstance(metrics, dict):
        return None
    return {
        "status": _string_or_default(result.get("status"), "partial"),
        "started_at_utc": _string_or_none(result.get("started_at_utc")),
        "closed_at_utc": _string_or_none(result.get("closed_at_utc")),
        "overall": _metric_group_payload(metrics.get("overall")),
        "courses": _metric_group_list_payload(metrics.get("courses")),
        "attempts": _attempts_payload(result.get("attempts")),
    }


def _metric_group_list_payload(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    return [metric for group in value if (metric := _metric_group_payload(group)) is not None]


def _metric_group_payload(value: object) -> dict[str, object] | None:
    if not isinstance(value, dict):
        return None
    primary = value.get("primary")
    detail = value.get("detail")
    if not isinstance(primary, dict) or not isinstance(detail, dict):
        return None
    return {
        "key": _string_or_default(value.get("key"), ""),
        "label": _string_or_default(value.get("label"), "Overall"),
        "attempt_count": _nonnegative_int(primary.get("attempt_count")),
        "success_count": _nonnegative_int(primary.get("success_count")),
        "success_rate": _number_or_none(primary.get("success_rate")),
        "finish_count": _nonnegative_int(primary.get("finish_count")),
        "finish_rate": _number_or_none(primary.get("finish_rate")),
        "completion_rate": _number_or_none(primary.get("completion_rate")),
        "mean_finish_time_ms": _number_or_none(primary.get("mean_finish_time_ms")),
        "best_finish_time_ms": _number_or_none(primary.get("best_finish_time_ms")),
        "mean_position": _number_or_none(primary.get("mean_position")),
        "best_position": _int_or_none(primary.get("best_position")),
        "total_env_steps": _nonnegative_int(primary.get("total_env_steps")),
        "mean_episode_length_steps": _number_or_none(primary.get("mean_episode_length_steps")),
        "mean_episode_return": _number_or_none(detail.get("mean_episode_return")),
        "best_episode_return": _number_or_none(detail.get("best_episode_return")),
        "average_speed": _number_or_none(detail.get("average_speed")),
    }


def _attempts_payload(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    return [_attempt_payload(attempt) for attempt in value if isinstance(attempt, dict)]


def _attempt_payload(attempt: dict[str, object]) -> dict[str, object]:
    course_result = _first_course_result(attempt.get("course_results"))
    return {
        "attempt_id": _string_or_default(attempt.get("attempt_id"), ""),
        "target_id": _string_or_default(attempt.get("target_id"), ""),
        "target_label": _string_or_none(attempt.get("target_label")),
        "status": _string_or_default(attempt.get("status"), "partial"),
        "seed": _int_string_or_none(attempt.get("seed")),
        "cup_id": _string_or_none(attempt.get("cup_id")),
        "difficulty": _string_or_none(attempt.get("difficulty")),
        "vehicle_id": _string_or_none(attempt.get("vehicle_id")),
        "total_race_time_ms": _int_or_none(attempt.get("total_race_time_ms")),
        "env_steps": _int_or_none(attempt.get("env_steps")),
        "episode_return": _number_or_none(attempt.get("episode_return")),
        "position": None if course_result is None else _int_or_none(course_result.get("position")),
        "completion_ratio": (
            None
            if course_result is None
            else _number_or_none(course_result.get("completion_ratio"))
        ),
        "closed_at_utc": _string_or_none(attempt.get("closed_at_utc")),
    }


def _first_course_result(value: object) -> dict[str, object] | None:
    if not isinstance(value, list) or not value:
        return None
    first = value[0]
    return first if isinstance(first, dict) else None


def _string_or_default(value: object, default: str) -> str:
    return value if isinstance(value, str) else default


def _string_or_none(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _int_or_none(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _int_string_or_none(value: object) -> str | None:
    integer = _int_or_none(value)
    return None if integer is None else str(integer)


def _nonnegative_int(value: object) -> int:
    integer = _int_or_none(value)
    return integer if integer is not None and integer >= 0 else 0


def _number_or_none(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)
