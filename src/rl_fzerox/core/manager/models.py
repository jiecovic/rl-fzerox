# src/rl_fzerox/core/manager/models.py
"""Dataclasses returned by the ManagerStore domain API."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from rl_fzerox.core.evaluation.models import (
    EvaluationCheckpointSnapshot,
    EvaluationPolicyMode,
    EvaluationTargetSpec,
)
from rl_fzerox.core.manager.run_spec import ManagedRunConfig

RunStatus = Literal["created", "running", "paused", "stopped", "finished", "failed", "archived"]
RunCommand = Literal["pause", "stop"]
SaveGameStatus = Literal["created", "running", "paused", "finished", "failed"]
SaveAttemptStatus = Literal["running", "succeeded", "failed"]
SaveUnlockInspectionStatus = Literal["not_inspected", "inspected"]
SaveUnlockTargetStatus = Literal["pending", "locked", "succeeded", "failed", "skipped"]
ViewerLeaseKind = Literal["run_watch", "career_mode"]
ManagedEvaluationStatus = Literal[
    "created",
    "running",
    "cancelling",
    "completed",
    "failed",
    "cancelled",
]
EvaluationBaselineSuiteStatus = Literal["not_created", "ready", "failed"]
PolicySourceKind = Literal["run", "evaluation", "checkpoint"]
PolicySourceArtifact = Literal["latest", "best", "final"]


@dataclass(frozen=True, slots=True)
class ManagedRunEvent:
    """One timestamped manager event for one run."""

    run_id: str
    created_at: str
    kind: str
    message: str


@dataclass(frozen=True, slots=True)
class ManagedRunRuntime:
    """Latest sampled runtime state for one managed run."""

    total_timesteps: int
    num_timesteps: int
    progress_fraction: float
    updated_at: str
    fps: float | None = None
    episode_reward_mean: float | None = None
    episode_length_mean: float | None = None
    approx_kl: float | None = None
    entropy_loss: float | None = None
    value_loss: float | None = None
    policy_gradient_loss: float | None = None


@dataclass(frozen=True, slots=True)
class ManagedRunMetricSample:
    """One historical metric sample captured for charts."""

    run_id: str
    created_at: str
    total_timesteps: int
    num_timesteps: int
    lineage_num_timesteps: int
    progress_fraction: float
    metrics: dict[str, float]
    fps: float | None = None
    episode_reward_mean: float | None = None
    episode_length_mean: float | None = None
    approx_kl: float | None = None
    entropy_loss: float | None = None
    value_loss: float | None = None
    policy_gradient_loss: float | None = None


@dataclass(frozen=True, slots=True)
class ManagedViewerLease:
    """One manager-owned visible viewer process lease."""

    id: str
    kind: ViewerLeaseKind
    owner_id: str
    pid: int
    launched_at: str
    heartbeat_at: str
    qualifier: str | None = None


@dataclass(frozen=True, slots=True)
class ManagedEvaluation:
    """One manager-owned evaluation record with immutable policy inputs."""

    id: str
    name: str
    status: ManagedEvaluationStatus
    evaluation_dir: Path
    source_run_id: str | None
    source_artifact: Literal["latest", "best", "final"] | None
    preset_id: str
    preset_version: int
    policy_mode: EvaluationPolicyMode
    seed: int
    target: EvaluationTargetSpec
    config: ManagedRunConfig
    checkpoint: EvaluationCheckpointSnapshot
    created_at: str
    updated_at: str
    started_at: str | None = None
    finished_at: str | None = None
    result_json_path: Path | None = None
    error_message: str | None = None


@dataclass(frozen=True, slots=True)
class ManagedEvaluationPreset:
    """One SQLite-owned evaluation preset version."""

    id: str
    name: str
    version: int
    seed: int
    renderer: Literal["angrylion", "gliden64"]
    target: EvaluationTargetSpec
    builtin: bool
    created_at: str
    updated_at: str


@dataclass(frozen=True, slots=True)
class ManagedEvaluationBaselineSuite:
    """Materialized reset-state suite for a preset version."""

    id: str
    preset_id: str
    preset_version: int
    status: EvaluationBaselineSuiteStatus
    suite_dir: Path
    manifest_path: Path | None = None
    error_message: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    materialized_at: str | None = None


@dataclass(frozen=True, slots=True)
class ManagedPublishedCheckpoint:
    """One installed published checkpoint bundle owned by the manager DB."""

    id: str
    checkpoint_id: str
    version: str
    name: str
    config: ManagedRunConfig
    config_hash: str
    import_dir: Path
    manifest_json: str
    source_bundle_path: Path | None
    source_bundle_sha256: str | None
    source_run_id: str | None
    source_run_name: str | None
    source_artifact: PolicySourceArtifact
    local_num_timesteps: int | None
    lineage_num_timesteps: int | None
    policy_path: Path
    model_path: Path
    checkpoint_metadata_path: Path
    train_config_path: Path
    evaluation_metrics_path: Path | None
    engine_tuning_state_path: Path | None
    engine_tuning_model_path: Path | None
    exported_at: str
    imported_at: str
    updated_at: str


@dataclass(frozen=True, slots=True)
class ManagedRun:
    """One immutable DB-managed run record."""

    id: str
    name: str
    status: RunStatus
    config: ManagedRunConfig
    config_hash: str
    run_dir: Path
    created_at: str
    lineage_id: str
    lineage_groups: tuple[str, ...] = ()
    lineage_step_offset: int = 0
    parent_run_id: str | None = None
    source_run_id: str | None = None
    source_artifact: Literal["latest", "best"] | None = None
    source_snapshot_dir: Path | None = None
    source_num_timesteps: int | None = None
    started_at: str | None = None
    stopped_at: str | None = None
    worker_heartbeat_at: str | None = None
    runtime: ManagedRunRuntime | None = None
    pending_command: RunCommand | None = None


@dataclass(frozen=True, slots=True)
class ManagedPolicySource:
    """One policy checkpoint source assignable to save-game course setups."""

    kind: PolicySourceKind
    id: str
    name: str
    artifact: PolicySourceArtifact
    config: ManagedRunConfig
    source_dir: Path
    mutable: bool
    created_at: str
    updated_at: str
    policy_path: Path | None = None
    model_path: Path | None = None
    engine_tuning_state_path: Path | None = None
    engine_tuning_model_path: Path | None = None
    source_run_id: str | None = None
    source_run_name: str | None = None
    local_num_timesteps: int | None = None
    lineage_num_timesteps: int | None = None


@dataclass(frozen=True, slots=True)
class ManagedRunVehicleSummary:
    """Vehicle settings needed when assigning a trained policy to a save game."""

    selection_mode: str
    selected_vehicle_ids: tuple[str, ...]
    engine_mode: str
    engine_setting_raw_value: int
    engine_setting_min_raw_value: int
    engine_setting_max_raw_value: int


@dataclass(frozen=True, slots=True)
class ManagedRunSummary:
    """Run-list record that omits the full config from API payloads."""

    id: str
    name: str
    status: RunStatus
    config_hash: str
    action_repeat: int
    vehicle_setup: ManagedRunVehicleSummary
    created_at: str
    lineage_id: str
    lineage_groups: tuple[str, ...] = ()
    lineage_step_offset: int = 0
    parent_run_id: str | None = None
    source_run_id: str | None = None
    source_artifact: Literal["latest", "best"] | None = None
    source_num_timesteps: int | None = None
    started_at: str | None = None
    stopped_at: str | None = None
    worker_heartbeat_at: str | None = None
    runtime: ManagedRunRuntime | None = None
    pending_command: RunCommand | None = None


@dataclass(frozen=True, slots=True)
class ManagedRunTemplate:
    """One DB-backed editable starting point for future immutable runs."""

    id: str
    name: str
    config: ManagedRunConfig
    config_hash: str
    created_at: str
    updated_at: str


@dataclass(frozen=True, slots=True)
class ManagedRunDraft:
    """One SQLite-only editable run configuration draft."""

    id: str
    name: str
    config: ManagedRunConfig
    config_hash: str
    created_at: str
    updated_at: str
    source_run_id: str | None = None
    source_artifact: Literal["latest", "best"] | None = None
    source_snapshot_dir: Path | None = None
    source_num_timesteps: int | None = None


@dataclass(frozen=True, slots=True)
class ManagedSaveGame:
    """One manager-owned portable save game and career-runner state root."""

    id: str
    name: str
    status: SaveGameStatus
    save_path: Path
    created_at: str
    updated_at: str
    last_finished_at: str | None = None
    runner_device: Literal["cpu", "cuda"] = "cuda"
    runner_renderer: Literal["angrylion", "gliden64"] = "gliden64"
    runner_policy_mode: Literal["deterministic", "stochastic"] = "deterministic"
    runner_attempt_seed: int | None = None
    runner_recording_enabled: bool = False
    runner_recording_input_hud_enabled: bool = False
    runner_recording_upscale_factor: int = 2
    runner_recording_path: Path | None = None
    runner_target_restart_on_retire: bool = False
    runner_target_clear_goal: int = 1
    runner_keep_failed_recordings: bool = False
    runner_reload_policy_between_attempts: bool = True


@dataclass(frozen=True, slots=True)
class ManagedSaveUnlockTarget:
    """One unlock-path target for a portable save game."""

    sequence_index: int
    kind: str
    status: SaveUnlockTargetStatus
    label: str
    difficulty: str | None = None
    cup_id: str | None = None
    course_id: str | None = None


@dataclass(frozen=True, slots=True)
class ManagedSaveUnlockProgress:
    """Progress for one save game, based on inspected save state when available."""

    inspection_status: SaveUnlockInspectionStatus
    completed_count: int
    total_count: int
    unlocked_vehicle_count: int
    unlocked_vehicle_ids: tuple[str, ...]
    next_target: ManagedSaveUnlockTarget | None
    targets: tuple[ManagedSaveUnlockTarget, ...]


@dataclass(frozen=True, slots=True)
class ManagedSaveCupSetup:
    """Machine setup selected once when a GP cup starts."""

    id: str
    save_game_id: str
    cup_id: str
    vehicle_id: str
    created_at: str
    updated_at: str
    difficulty: str | None = None


@dataclass(frozen=True, slots=True)
class ManagedSaveCourseSetup:
    """Policy and engine setup for one career course."""

    id: str
    save_game_id: str
    policy_source_kind: PolicySourceKind
    policy_source_id: str
    policy_artifact: PolicySourceArtifact
    engine_setting_raw_value: int
    created_at: str
    updated_at: str
    difficulty: str | None = None
    cup_id: str | None = None
    course_id: str | None = None


@dataclass(frozen=True, slots=True)
class ManagedSaveAttempt:
    """One concrete policy attempt for one unlock-path target."""

    id: str
    save_game_id: str
    status: SaveAttemptStatus
    started_at: str
    target_kind: str | None = None
    difficulty: str | None = None
    cup_id: str | None = None
    course_id: str | None = None
    finished_at: str | None = None
    finish_position: int | None = None
    finish_time_s: float | None = None
    failure_reason: str | None = None
