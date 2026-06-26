# src/rl_fzerox/apps/run_manager/api/contracts.py
from __future__ import annotations

from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rl_fzerox.core.domain.engine import (
    ENGINE_SLIDER,
    engine_percent_to_slider_step,
)
from rl_fzerox.core.evaluation.models import (
    EVALUATION_TARGET_LIMITS,
    EvaluationCheckpointArtifact,
    EvaluationMode,
)
from rl_fzerox.core.manager import ManagedRun, ManagedRunConfig
from rl_fzerox.core.runtime_spec.renderers import RendererName

WatchDevice = Literal["cpu", "cuda"]
type WatchRenderer = RendererName
PolicyPlaybackMode = Literal["deterministic", "stochastic"]
EngineTuningSourceAction = Literal["convert", "discard"]
PolicySourceKind = Literal["run", "evaluation", "checkpoint"]
PolicySourceArtifact = Literal["latest", "best", "final"]


class EvaluationTargetRequest(BaseModel):
    """Request body section for one evaluation target set."""

    model_config = ConfigDict(extra="forbid")

    mode: EvaluationMode = "time_attack_course"
    course_ids: tuple[str, ...] = ()
    cup_ids: tuple[str, ...] = ()
    difficulties: tuple[str, ...] = ()
    repeats_per_target: int = Field(default=1, ge=1, le=1000)
    baseline_variant_count: int = Field(
        default=1,
        ge=1,
        le=EVALUATION_TARGET_LIMITS.baseline_variant_count,
    )

    @model_validator(mode="after")
    def _validate_mode_specific_target(self) -> EvaluationTargetRequest:
        if self.mode == "gp_course":
            if len(self.difficulties) != 1:
                raise ValueError("gp_course evaluation presets require exactly one difficulty")
            return self
        if self.difficulties:
            raise ValueError("time_attack_course evaluation presets must not set difficulties")
        if self.baseline_variant_count != 1:
            raise ValueError("time_attack_course evaluation presets must use one baseline variant")
        return self


class CreateEvaluationRequest(BaseModel):
    """Request body for creating one immutable evaluation snapshot."""

    model_config = ConfigDict(extra="forbid")

    name: str
    source_policy_kind: PolicySourceKind = "run"
    source_policy_id: str | None = None
    source_run_id: str | None = None
    source_artifact: EvaluationCheckpointArtifact = "latest"
    preset_id: str
    policy_mode: PolicyPlaybackMode = "deterministic"

    @model_validator(mode="after")
    def _validate_source(self) -> CreateEvaluationRequest:
        if self.source_policy_id is not None:
            if self.source_run_id is not None and self.source_policy_kind != "run":
                raise ValueError(
                    "source_run_id is only valid with source_policy_kind='run'"
                )
            return self
        if self.source_policy_kind == "run" and self.source_run_id is not None:
            return self
        raise ValueError("source_policy_id is required")


class StartEvaluationRequest(BaseModel):
    """Runtime options for launching one evaluation worker."""

    model_config = ConfigDict(extra="forbid")

    device: WatchDevice = "cuda"
    worker_count: int = Field(default=1, ge=1, le=32)


class CreateEvaluationPresetRequest(BaseModel):
    """Request body for creating one immutable evaluation preset."""

    model_config = ConfigDict(extra="forbid")

    name: str
    seed: int = Field(ge=0, le=(1 << 32) - 1)
    renderer: WatchRenderer = "gliden64"
    target: EvaluationTargetRequest = Field(default_factory=EvaluationTargetRequest)


class UpdateEvaluationRequest(BaseModel):
    """Request body for renaming one manager-owned evaluation."""

    model_config = ConfigDict(extra="forbid")

    name: str


class CreateDraftRequest(BaseModel):
    """Request body for creating one SQLite-backed draft."""

    model_config = ConfigDict(extra="forbid")

    name: str
    config: ManagedRunConfig
    source_policy_kind: PolicySourceKind | None = None
    source_policy_id: str | None = None
    source_run_id: str | None = None
    source_artifact: Literal["latest", "best"] | None = None


class CreateSaveGameRequest(BaseModel):
    """Request body for creating one managed career-runner save game."""

    model_config = ConfigDict(extra="forbid")

    name: str


class StartCareerModeRequest(BaseModel):
    """Request body for launching one Career Mode runner."""

    model_config = ConfigDict(extra="forbid")

    device: WatchDevice = "cuda"
    renderer: WatchRenderer | None = None
    attempt_seed: int | None = Field(default=None, ge=0, le=(1 << 32) - 1)
    policy_mode: PolicyPlaybackMode = "deterministic"
    recording_enabled: bool = False
    recording_input_hud_enabled: bool = False
    recording_upscale_factor: int = Field(default=2, ge=1, le=4)
    recording_path: Path | None = None
    target_kind: str | None = None
    difficulty: str | None = None
    cup_id: str | None = None
    course_id: str | None = None
    single_target: bool = False
    perfect_run: bool = False
    keep_failed_recordings: bool = True
    target_clear_goal: int = Field(default=0, ge=0)
    reload_policy_between_attempts: bool = True

    @model_validator(mode="after")
    def _validate_target_fields(self) -> StartCareerModeRequest:
        if not self.recording_enabled and (
            self.target_clear_goal > 0 or not self.keep_failed_recordings
        ):
            raise ValueError("target recording options require recording_enabled=true")
        values = (self.target_kind, self.difficulty, self.cup_id, self.course_id)
        if not any(value is not None for value in values):
            if self.perfect_run or self.target_clear_goal > 0 or not self.keep_failed_recordings:
                raise ValueError("target fishing options require a selected single target")
            return self
        if self.target_kind is None or self.difficulty is None or self.cup_id is None:
            raise ValueError("target_kind, difficulty, and cup_id are required together")
        if (
            self.perfect_run or self.target_clear_goal > 0 or not self.keep_failed_recordings
        ) and not self.single_target:
            raise ValueError("target fishing options require single_target=true")
        return self


class UpdateSaveGameRequest(BaseModel):
    """Request body for renaming one manager-owned save game."""

    model_config = ConfigDict(extra="forbid")

    name: str


class UpdateSaveRunnerSettingsRequest(BaseModel):
    """Request body for saved Career Mode runner launch settings."""

    model_config = ConfigDict(extra="forbid")

    device: WatchDevice = "cuda"
    renderer: WatchRenderer = "gliden64"
    attempt_seed: int | None = Field(default=None, ge=0, le=(1 << 32) - 1)
    policy_mode: PolicyPlaybackMode = "deterministic"
    recording_enabled: bool = False
    recording_input_hud_enabled: bool = False
    recording_upscale_factor: int = Field(default=2, ge=1, le=4)
    recording_path: Path | None = None
    target_restart_on_retire: bool = False
    target_clear_goal: int = Field(default=1, ge=0)
    keep_failed_recordings: bool = False
    reload_policy_between_attempts: bool = True


class UpsertSaveCourseSetupRequest(BaseModel):
    """Request body for assigning a policy and engine to one save-game course."""

    model_config = ConfigDict(extra="forbid")

    policy_source_kind: PolicySourceKind = "run"
    policy_source_id: str
    policy_artifact: PolicySourceArtifact = "best"
    engine_setting_raw_value: int = Field(
        default=engine_percent_to_slider_step(50),
        ge=ENGINE_SLIDER.min_step,
        le=ENGINE_SLIDER.max_step,
    )
    difficulty: str | None = None
    cup_id: str | None = None
    course_id: str | None = None


class UpsertSaveCupSetupRequest(BaseModel):
    """Request body for assigning a vehicle to one save-game cup."""

    model_config = ConfigDict(extra="forbid")

    cup_id: str
    vehicle_id: str = "blue_falcon"
    difficulty: str | None = None


class ImportSaveEngineTuningCourseSetupRequest(BaseModel):
    """One draft course setup that should receive an engine recommendation."""

    model_config = ConfigDict(extra="forbid")

    difficulty: str | None = None
    cup_id: str
    course_id: str
    vehicle_id: str


class ImportSaveEngineTuningRequest(BaseModel):
    """Request body for learned engine-setting recommendations."""

    model_config = ConfigDict(extra="forbid")

    course_setups: tuple[ImportSaveEngineTuningCourseSetupRequest, ...] = ()
    policy_source_kind: PolicySourceKind = "run"
    policy_source_id: str
    policy_artifact: PolicySourceArtifact = "latest"


class UpdateDraftRequest(BaseModel):
    """Request body for updating one SQLite-backed draft."""

    model_config = ConfigDict(extra="forbid")

    name: str
    config: ManagedRunConfig
    source_policy_kind: PolicySourceKind | None = None
    source_policy_id: str | None = None
    source_run_id: str | None = None
    source_artifact: Literal["latest", "best"] | None = None


class UpdateRunRequest(BaseModel):
    """Request body for renaming one managed training run."""

    model_config = ConfigDict(extra="forbid")

    name: str


class WatchRunRequest(BaseModel):
    """Request body for launching one run in the watch app."""

    model_config = ConfigDict(extra="forbid")

    device: WatchDevice = "cuda"
    renderer: WatchRenderer | None = None
    policy_mode: PolicyPlaybackMode = "deterministic"


class UpdateLineageGroupsRequest(BaseModel):
    """Request body for assigning one lineage to UI/TensorBoard groups."""

    model_config = ConfigDict(extra="forbid")

    group_names: tuple[str, ...] = ()


class LaunchRunRequest(BaseModel):
    """Request body for launching one managed training run."""

    model_config = ConfigDict(extra="forbid")

    name: str
    config: ManagedRunConfig
    draft_id: str | None = None
    source_policy_kind: PolicySourceKind | None = None
    source_policy_id: str | None = None
    source_run_id: str | None = None
    source_artifact: Literal["latest", "best"] | None = None
    copy_alt_baselines: bool = True
    engine_tuning_source_action: EngineTuningSourceAction = "convert"


class ForkRunRequest(BaseModel):
    """Request body for forking one managed training run."""

    model_config = ConfigDict(extra="forbid")

    artifact: Literal["latest", "best"]
    name: str | None = None
    config: ManagedRunConfig | None = None
    copy_alt_baselines: bool = True
    engine_tuning_source_action: EngineTuningSourceAction = "convert"


class RunLauncher(Protocol):
    def launch(
        self,
        *,
        name: str,
        config: ManagedRunConfig,
        draft_id: str | None,
        source_policy_kind: PolicySourceKind | None,
        source_policy_id: str | None,
        source_run_id: str | None,
        source_artifact: Literal["latest", "best"] | None,
        copy_alt_baselines: bool,
        engine_tuning_source_action: EngineTuningSourceAction,
    ) -> ManagedRun: ...

    def fork(
        self,
        *,
        run_id: str,
        artifact: Literal["latest", "best"],
        name: str | None,
        config: ManagedRunConfig | None,
        copy_alt_baselines: bool,
        engine_tuning_source_action: EngineTuningSourceAction,
    ) -> ManagedRun: ...

    def request_pause(self, *, run_id: str) -> ManagedRun: ...

    def request_stop(self, *, run_id: str) -> ManagedRun: ...

    def resume(self, *, run_id: str) -> ManagedRun: ...

    def watch_artifact(
        self,
        *,
        run_id: str,
        artifact: str,
        device: WatchDevice,
        renderer: WatchRenderer | None,
        deterministic_policy: bool,
    ) -> Literal["started", "already_running"]: ...

    def start_career_mode(
        self,
        *,
        save_game_id: str,
        device: WatchDevice,
        renderer: WatchRenderer | None,
        attempt_seed: int | None,
        deterministic_policy: bool,
        recording_enabled: bool,
        recording_input_hud_enabled: bool,
        recording_upscale_factor: int,
        recording_path: Path | None,
        target_kind: str | None,
        difficulty: str | None,
        cup_id: str | None,
        course_id: str | None,
        single_target: bool,
        perfect_run: bool,
        keep_failed_recordings: bool,
        target_clear_goal: int,
        reload_policy_between_attempts: bool,
    ) -> Literal["started", "already_running"]: ...
