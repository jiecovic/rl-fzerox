# src/rl_fzerox/core/evaluation/models.py
"""Immutable evaluation specs, targets, runtime settings, and results.

These dataclasses are the durable contract between manager records, execution,
reporting artifacts, and frontend payloads. Their validation is intentionally
limited to stable shape invariants; runtime semantics live in the owning runner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

type EvaluationMode = Literal["time_attack_course", "gp_course"]
type EvaluationCheckpointArtifact = Literal["latest", "best"]
type EvaluationPolicyMode = Literal["deterministic", "stochastic"]
type EvaluationDevice = Literal["cpu", "cuda"]
type EvaluationRunStatus = Literal["completed", "failed", "cancelled", "partial"]
type AttemptStatus = Literal["succeeded", "failed", "cancelled", "partial"]
type CourseResultStatus = Literal["finished", "retired", "crashed", "truncated", "failed"]


@dataclass(frozen=True, slots=True)
class EvaluationTargetLimits:
    """Validation limits shared by evaluation API and persistence boundaries."""

    baseline_variant_count: int = 16


EVALUATION_TARGET_LIMITS = EvaluationTargetLimits()


def _require_non_empty_text(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _require_text_items(values: tuple[str, ...], field_name: str) -> None:
    for index, value in enumerate(values):
        _require_non_empty_text(value, f"{field_name}[{index}]")


def _require_minimum_int(value: object, field_name: str, minimum: int) -> None:
    if not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field_name} must be at least {minimum}")


def _require_maximum_int(value: int, field_name: str, maximum: int) -> None:
    if value > maximum:
        raise ValueError(f"{field_name} must be at most {maximum}")


def _require_optional_minimum_int(
    value: int | None,
    field_name: str,
    minimum: int,
) -> None:
    if value is not None:
        _require_minimum_int(value, field_name, minimum)


def _require_optional_non_empty_text(value: str | None, field_name: str) -> None:
    if value is not None:
        _require_non_empty_text(value, field_name)


@dataclass(frozen=True, slots=True)
class EvaluationTargetSpec:
    """Declared target set for one evaluation run."""

    mode: EvaluationMode
    course_ids: tuple[str, ...] = ()
    cup_ids: tuple[str, ...] = ()
    difficulties: tuple[str, ...] = ()
    vehicle_ids: tuple[str, ...] = ()
    repeats_per_target: int = 1
    baseline_variant_count: int = 1

    def __post_init__(self) -> None:
        _require_text_items(self.course_ids, "course_ids")
        _require_text_items(self.cup_ids, "cup_ids")
        _require_text_items(self.difficulties, "difficulties")
        _require_text_items(self.vehicle_ids, "vehicle_ids")
        _require_minimum_int(self.repeats_per_target, "repeats_per_target", 1)
        _require_minimum_int(self.baseline_variant_count, "baseline_variant_count", 1)
        _require_maximum_int(
            self.baseline_variant_count,
            "baseline_variant_count",
            EVALUATION_TARGET_LIMITS.baseline_variant_count,
        )


@dataclass(frozen=True, slots=True)
class EvaluationCourseTarget:
    """One concrete single-course evaluation target.

    Higher-level specs can expand into this form before execution. The runner
    keeps this as data instead of reading manager state so evaluation attempts
    remain reproducible from their written summary.
    """

    target_id: str
    course_id: str
    course_name: str | None = None
    cup_id: str | None = None
    difficulty: str | None = None
    vehicle_id: str | None = None
    baseline_state_path: str | None = None
    baseline_group_id: str | None = None
    baseline_variant_index: int | None = None
    baseline_variant_count: int | None = None
    baseline_variant_seed: int | None = None
    engine_setting_raw_value: int | None = None

    def __post_init__(self) -> None:
        _require_non_empty_text(self.target_id, "target_id")
        _require_non_empty_text(self.course_id, "course_id")
        _require_optional_non_empty_text(self.course_name, "course_name")
        _require_optional_non_empty_text(self.cup_id, "cup_id")
        _require_optional_non_empty_text(self.difficulty, "difficulty")
        _require_optional_non_empty_text(self.vehicle_id, "vehicle_id")
        _require_optional_non_empty_text(self.baseline_state_path, "baseline_state_path")
        _require_optional_non_empty_text(self.baseline_group_id, "baseline_group_id")
        _require_optional_minimum_int(
            self.baseline_variant_index,
            "baseline_variant_index",
            0,
        )
        _require_optional_minimum_int(
            self.baseline_variant_count,
            "baseline_variant_count",
            1,
        )
        _require_optional_minimum_int(self.baseline_variant_seed, "baseline_variant_seed", 0)


@dataclass(frozen=True, slots=True)
class EvaluationCheckpointSnapshot:
    """Immutable checkpoint copy evaluated by a run.

    The source path may be a moving artifact like ``latest.zip``. The copied
    checkpoint path must point at the frozen file inside the evaluation artifact
    directory so later training checkpoint writes cannot change the result.
    """

    source_run_id: str | None
    source_run_name: str | None
    artifact: EvaluationCheckpointArtifact
    source_policy_path: str
    copied_policy_path: str
    source_model_path: str | None = None
    copied_model_path: str | None = None
    local_num_timesteps: int | None = None
    lineage_num_timesteps: int | None = None
    source_mtime_ns: int | None = None

    def __post_init__(self) -> None:
        _require_non_empty_text(self.source_policy_path, "source_policy_path")
        _require_non_empty_text(self.copied_policy_path, "copied_policy_path")
        _require_optional_non_empty_text(self.source_model_path, "source_model_path")
        _require_optional_non_empty_text(self.copied_model_path, "copied_model_path")
        _require_optional_minimum_int(self.local_num_timesteps, "local_num_timesteps", 0)
        _require_optional_minimum_int(
            self.lineage_num_timesteps,
            "lineage_num_timesteps",
            0,
        )
        _require_optional_minimum_int(self.source_mtime_ns, "source_mtime_ns", 0)


@dataclass(frozen=True, slots=True)
class EvaluationSpec:
    """Frozen configuration for a reproducible evaluation run."""

    evaluation_id: str
    seed: int
    target: EvaluationTargetSpec
    checkpoint: EvaluationCheckpointSnapshot
    policy_mode: EvaluationPolicyMode = "deterministic"
    total_planned_attempts: int | None = None

    def __post_init__(self) -> None:
        _require_non_empty_text(self.evaluation_id, "evaluation_id")
        _require_minimum_int(self.seed, "seed", 0)
        _require_optional_minimum_int(
            self.total_planned_attempts,
            "total_planned_attempts",
            1,
        )


@dataclass(frozen=True, slots=True)
class EvaluationRuntimeSpec:
    """Runtime settings used to execute one evaluation attempt plan."""

    device: EvaluationDevice = "cuda"
    worker_count: int = 1

    def __post_init__(self) -> None:
        _require_minimum_int(self.worker_count, "worker_count", 1)


@dataclass(frozen=True, slots=True)
class EvaluationCourseResult:
    """Terminal result for one course episode or one GP course attempt."""

    course_id: str
    status: CourseResultStatus
    course_name: str | None = None
    cup_id: str | None = None
    difficulty: str | None = None
    vehicle_id: str | None = None
    seed: int | None = None
    race_time_ms: int | None = None
    position: int | None = None
    completion_ratio: float | None = None
    laps_completed: int | None = None
    total_laps: int | None = None
    env_steps: int | None = None
    episode_length_steps: int | None = None
    episode_return: float | None = None
    engine_setting_raw_value: int | None = None
    ko_stars: int | None = None
    failure_reason: str | None = None
    boost_pad_entries: int | None = None
    average_speed: float | None = None


@dataclass(frozen=True, slots=True)
class EvaluationAttemptResult:
    """One repeated single-course target attempt."""

    attempt_id: str
    target_id: str
    status: AttemptStatus
    target_label: str | None = None
    cup_id: str | None = None
    difficulty: str | None = None
    vehicle_id: str | None = None
    seed: int | None = None
    started_at_utc: str | None = None
    closed_at_utc: str | None = None
    total_race_time_ms: int | None = None
    env_steps: int | None = None
    episode_length_steps: int | None = None
    episode_return: float | None = None
    course_results: tuple[EvaluationCourseResult, ...] = ()


@dataclass(frozen=True, slots=True)
class EvaluationRunResult:
    """Complete or partial result for one evaluation run."""

    spec: EvaluationSpec
    status: EvaluationRunStatus
    runtime: EvaluationRuntimeSpec = field(default_factory=EvaluationRuntimeSpec)
    started_at_utc: str | None = None
    closed_at_utc: str | None = None
    attempts: tuple[EvaluationAttemptResult, ...] = field(default_factory=tuple)
