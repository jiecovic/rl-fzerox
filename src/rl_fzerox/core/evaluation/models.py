# src/rl_fzerox/core/evaluation/models.py
"""Immutable evaluation specs and result records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TypeAlias

EvaluationMode: TypeAlias = Literal["time_attack", "gp_cup"]
EvaluationCheckpointArtifact: TypeAlias = Literal["latest", "best", "final"]
EvaluationPolicyMode: TypeAlias = Literal["deterministic", "stochastic"]
EvaluationRunStatus: TypeAlias = Literal["completed", "failed", "cancelled", "partial"]
AttemptStatus: TypeAlias = Literal["succeeded", "failed", "cancelled", "partial"]
CourseResultStatus: TypeAlias = Literal["finished", "retired", "crashed", "truncated", "failed"]


@dataclass(frozen=True, slots=True)
class EvaluationTargetSpec:
    """Declared target set for one evaluation run."""

    mode: EvaluationMode
    course_ids: tuple[str, ...] = ()
    cup_ids: tuple[str, ...] = ()
    difficulties: tuple[str, ...] = ()
    vehicle_ids: tuple[str, ...] = ()
    repeats_per_target: int = 1


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
    engine_setting_raw_value: int | None = None


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


@dataclass(frozen=True, slots=True)
class EvaluationSpec:
    """Frozen configuration for a reproducible evaluation run."""

    evaluation_id: str
    seed: int
    target: EvaluationTargetSpec
    checkpoint: EvaluationCheckpointSnapshot
    policy_mode: EvaluationPolicyMode = "deterministic"
    total_planned_attempts: int | None = None


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
    boost_active_count: int | None = None
    boost_active_frames: int | None = None
    boost_pad_entries: int | None = None
    damage_event_count: int | None = None
    minimum_height: float | None = None
    average_speed: float | None = None


@dataclass(frozen=True, slots=True)
class EvaluationAttemptResult:
    """One repeated target attempt.

    For single-course suites this usually contains one course result. For GP and
    Career targets it can contain multiple course results and GP-level final
    rank/points fields.
    """

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
    final_gp_position: int | None = None
    gp_points: int | None = None
    total_race_time_ms: int | None = None
    env_steps: int | None = None
    episode_length_steps: int | None = None
    episode_return: float | None = None
    course_results: tuple[EvaluationCourseResult, ...] = ()
    artifact_paths: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class EvaluationRunResult:
    """Complete or partial result for one evaluation run."""

    spec: EvaluationSpec
    status: EvaluationRunStatus
    started_at_utc: str | None = None
    closed_at_utc: str | None = None
    attempts: tuple[EvaluationAttemptResult, ...] = field(default_factory=tuple)
