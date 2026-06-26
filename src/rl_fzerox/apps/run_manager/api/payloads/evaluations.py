# src/rl_fzerox/apps/run_manager/api/payloads/evaluations.py
from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Literal

from typing_extensions import TypedDict

from rl_fzerox.core.evaluation.models import (
    EvaluationCheckpointArtifact,
    EvaluationCheckpointSnapshot,
    EvaluationDevice,
    EvaluationPolicyMode,
    EvaluationTargetSpec,
)
from rl_fzerox.core.manager import (
    ManagedEvaluation,
    ManagedEvaluationBaselineSuite,
    ManagedEvaluationPreset,
)
from rl_fzerox.core.manager.models import (
    EvaluationBaselineSuiteStatus,
    ManagedEvaluationStatus,
)

JsonObject = Mapping[str, object]


class EvaluationTargetPayload(TypedDict):
    mode: str
    course_ids: list[str]
    cup_ids: list[str]
    difficulties: list[str]
    vehicle_ids: list[str]
    repeats_per_target: int
    baseline_variant_count: int


class EvaluationProgressPayload(TypedDict):
    completed_attempts: int
    total_attempts: int | None
    result_status: str | None


class EvaluationCheckpointPayload(TypedDict):
    source_run_id: str | None
    source_run_name: str | None
    artifact: EvaluationCheckpointArtifact
    source_policy_path: str
    copied_policy_path: str
    source_model_path: str | None
    copied_model_path: str | None
    local_num_timesteps: int | None
    lineage_num_timesteps: int | None
    source_mtime_ns: str | None


class EvaluationMetricSummaryPayload(TypedDict):
    key: str
    label: str
    attempt_count: int
    success_count: int
    success_rate: float | None
    finish_count: int
    finish_rate: float | None
    completion_rate: float | None
    mean_finish_time_ms: float | None
    best_finish_time_ms: float | None
    mean_position: float | None
    best_position: int | None
    total_env_steps: int
    mean_episode_length_steps: float | None
    mean_episode_return: float | None
    best_episode_return: float | None
    average_speed: float | None


class EvaluationAttemptSummaryPayload(TypedDict):
    attempt_id: str
    target_id: str
    target_label: str | None
    status: str
    seed: str | None
    cup_id: str | None
    difficulty: str | None
    vehicle_id: str | None
    total_race_time_ms: int | None
    env_steps: int | None
    episode_return: float | None
    position: int | None
    completion_ratio: float | None
    closed_at_utc: str | None


class EvaluationRuntimePayload(TypedDict):
    device: EvaluationDevice
    worker_count: int


class EvaluationResultSummaryPayload(TypedDict):
    status: str
    started_at_utc: str | None
    closed_at_utc: str | None
    runtime: EvaluationRuntimePayload | None
    overall: EvaluationMetricSummaryPayload | None
    courses: list[EvaluationMetricSummaryPayload]
    attempts: list[EvaluationAttemptSummaryPayload]


class EvaluationBaselineSuitePayload(TypedDict):
    id: str
    preset_id: str
    preset_version: int
    status: EvaluationBaselineSuiteStatus
    suite_dir: str
    manifest_path: str | None
    error_message: str | None
    created_at: str | None
    updated_at: str | None
    materialized_at: str | None


class EvaluationPayload(TypedDict):
    id: str
    name: str
    status: ManagedEvaluationStatus
    evaluation_dir: str
    source_run_id: str | None
    source_artifact: EvaluationCheckpointArtifact | None
    preset_id: str
    preset_version: int
    policy_mode: EvaluationPolicyMode
    seed: int
    target: EvaluationTargetPayload
    config: dict[str, object]
    checkpoint: EvaluationCheckpointPayload
    created_at: str
    updated_at: str
    started_at: str | None
    finished_at: str | None
    result_json_path: str | None
    error_message: str | None
    progress: EvaluationProgressPayload
    result_summary: EvaluationResultSummaryPayload | None
    baseline_suite: EvaluationBaselineSuitePayload


class EvaluationPresetPayload(TypedDict):
    id: str
    name: str
    version: int
    seed: int
    renderer: Literal["angrylion", "gliden64"]
    target: EvaluationTargetPayload
    builtin: bool
    created_at: str
    updated_at: str


class EvaluationsPayload(TypedDict):
    evaluations: list[EvaluationPayload]
    presets: list[EvaluationPresetPayload]
    baseline_suites: list[EvaluationBaselineSuitePayload]


class EvaluationResponsePayload(TypedDict):
    evaluation: EvaluationPayload


class EvaluationPresetResponsePayload(TypedDict):
    preset: EvaluationPresetPayload


class DeleteResponsePayload(TypedDict):
    deleted: bool


def evaluation_payload(
    evaluation: ManagedEvaluation,
    *,
    baseline_suite: ManagedEvaluationBaselineSuite,
) -> EvaluationPayload:
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
        "seed": evaluation.seed,
        "target": _target_payload(evaluation.target),
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


def evaluation_preset_payload(preset: ManagedEvaluationPreset) -> EvaluationPresetPayload:
    """Return one persisted evaluation-preset payload."""

    return {
        "id": preset.id,
        "name": preset.name,
        "version": preset.version,
        "seed": preset.seed,
        "renderer": preset.renderer,
        "target": _target_payload(preset.target),
        "builtin": preset.builtin,
        "created_at": preset.created_at,
        "updated_at": preset.updated_at,
    }


def evaluation_baseline_suite_payload(
    suite: ManagedEvaluationBaselineSuite,
) -> EvaluationBaselineSuitePayload:
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


def _target_payload(target: EvaluationTargetSpec) -> EvaluationTargetPayload:
    return {
        "mode": target.mode,
        "course_ids": list(target.course_ids),
        "cup_ids": list(target.cup_ids),
        "difficulties": list(target.difficulties),
        "vehicle_ids": list(target.vehicle_ids),
        "repeats_per_target": target.repeats_per_target,
        "baseline_variant_count": target.baseline_variant_count,
    }


def _checkpoint_payload(checkpoint: EvaluationCheckpointSnapshot) -> EvaluationCheckpointPayload:
    # Nanosecond mtimes exceed JavaScript's safe integer range. Keep the Python
    # model and persisted spec numeric, but expose the API value losslessly.
    return {
        "source_run_id": checkpoint.source_run_id,
        "source_run_name": checkpoint.source_run_name,
        "artifact": checkpoint.artifact,
        "source_policy_path": checkpoint.source_policy_path,
        "copied_policy_path": checkpoint.copied_policy_path,
        "source_model_path": checkpoint.source_model_path,
        "copied_model_path": checkpoint.copied_model_path,
        "local_num_timesteps": checkpoint.local_num_timesteps,
        "lineage_num_timesteps": checkpoint.lineage_num_timesteps,
        "source_mtime_ns": (
            None if checkpoint.source_mtime_ns is None else str(checkpoint.source_mtime_ns)
        ),
    }


def _read_result_file(evaluation: ManagedEvaluation) -> dict[str, object] | None:
    if evaluation.result_json_path is None or not evaluation.result_json_path.is_file():
        return None
    try:
        payload: object = json.loads(evaluation.result_json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return _object_mapping(payload)


def _progress_payload(payload: JsonObject | None) -> EvaluationProgressPayload:
    completed_attempts = 0
    total_attempts: int | None = None
    result_status: str | None = None
    result = _object_mapping(payload.get("result")) if payload is not None else None
    if result is not None:
        attempts = result.get("attempts")
        if isinstance(attempts, list):
            completed_attempts = len(attempts)
        spec = _object_mapping(result.get("spec"))
        if spec is not None:
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


def _result_summary_payload(payload: JsonObject | None) -> EvaluationResultSummaryPayload | None:
    result = _object_mapping(payload.get("result")) if payload is not None else None
    metrics = _object_mapping(payload.get("metrics")) if payload is not None else None
    if result is None or metrics is None:
        return None
    return {
        "status": _string_or_default(result.get("status"), "partial"),
        "started_at_utc": _string_or_none(result.get("started_at_utc")),
        "closed_at_utc": _string_or_none(result.get("closed_at_utc")),
        "runtime": _runtime_payload(result.get("runtime")),
        "overall": _metric_group_payload(metrics.get("overall")),
        "courses": _metric_group_list_payload(metrics.get("courses")),
        "attempts": _attempts_payload(result.get("attempts")),
    }


def _metric_group_list_payload(value: object) -> list[EvaluationMetricSummaryPayload]:
    if not isinstance(value, list):
        return []
    return [metric for group in value if (metric := _metric_group_payload(group)) is not None]


def _metric_group_payload(value: object) -> EvaluationMetricSummaryPayload | None:
    group = _object_mapping(value)
    if group is None:
        return None
    primary = _object_mapping(group.get("primary"))
    detail = _object_mapping(group.get("detail"))
    if primary is None or detail is None:
        return None
    return {
        "key": _string_or_default(group.get("key"), ""),
        "label": _string_or_default(group.get("label"), "Overall"),
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


def _attempts_payload(value: object) -> list[EvaluationAttemptSummaryPayload]:
    if not isinstance(value, list):
        return []
    return [
        payload
        for attempt in value
        if (payload := _attempt_payload(_object_mapping(attempt))) is not None
    ]


def _attempt_payload(attempt: JsonObject | None) -> EvaluationAttemptSummaryPayload | None:
    if attempt is None:
        return None
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
        "position": _attempt_position(attempt, course_result),
        "completion_ratio": (
            None
            if course_result is None
            else _number_or_none(course_result.get("completion_ratio"))
        ),
        "closed_at_utc": _string_or_none(attempt.get("closed_at_utc")),
    }


def _runtime_payload(value: object) -> EvaluationRuntimePayload | None:
    runtime = _object_mapping(value)
    if runtime is None:
        return None
    return {
        "device": _device_or_default(runtime.get("device")),
        "worker_count": _worker_count_or_default(runtime.get("worker_count")),
    }


def _first_course_result(value: object) -> dict[str, object] | None:
    if not isinstance(value, list) or not value:
        return None
    return _object_mapping(value[0])


def _attempt_position(
    attempt: JsonObject,
    course_result: JsonObject | None,
) -> int | None:
    if course_result is None:
        return None
    position = _int_or_none(course_result.get("position"))
    if position is None:
        return None
    status = _string_or_none(course_result.get("status")) or _string_or_none(attempt.get("status"))
    if status in {"finished", "succeeded"}:
        return position
    return max(position, 30)


def _object_mapping(value: object) -> dict[str, object] | None:
    if not isinstance(value, dict):
        return None
    mapping: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            return None
        mapping[key] = item
    return mapping


def _string_or_default(value: object, default: str) -> str:
    return value if isinstance(value, str) else default


def _string_or_none(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _device_or_default(value: object) -> EvaluationDevice:
    if value == "cpu":
        return "cpu"
    if value == "cuda":
        return "cuda"
    return "cuda"


def _worker_count_or_default(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return 1
    return value if value >= 1 else 1


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
