# src/rl_fzerox/apps/run_manager/api/contracts.py
from __future__ import annotations

from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rl_fzerox.core.manager import ManagedRun, ManagedRunConfig

WatchDevice = Literal["cpu", "cuda"]
WatchRenderer = Literal["angrylion", "gliden64"]
PolicyPlaybackMode = Literal["deterministic", "stochastic"]
EngineTuningSourceAction = Literal["convert", "discard"]


class CreateDraftRequest(BaseModel):
    """Request body for creating one SQLite-backed draft."""

    model_config = ConfigDict(extra="forbid")

    name: str
    config: ManagedRunConfig
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
    recording_path: Path | None = None
    target_kind: str | None = None
    difficulty: str | None = None
    cup_id: str | None = None
    course_id: str | None = None
    single_target: bool = False

    @model_validator(mode="after")
    def _validate_target_fields(self) -> StartCareerModeRequest:
        if self.recording_enabled and self.recording_path is None:
            raise ValueError("recording_path is required when recording_enabled is true")
        values = (self.target_kind, self.difficulty, self.cup_id, self.course_id)
        if not any(value is not None for value in values):
            return self
        if self.target_kind is None or self.difficulty is None or self.cup_id is None:
            raise ValueError("target_kind, difficulty, and cup_id are required together")
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
    recording_path: Path | None = None

    @model_validator(mode="after")
    def _validate_recording_path(self) -> UpdateSaveRunnerSettingsRequest:
        if self.recording_enabled and self.recording_path is None:
            raise ValueError("recording_path is required when recording_enabled is true")
        return self


class UpsertSaveCourseSetupRequest(BaseModel):
    """Request body for assigning a policy and engine to one save-game course."""

    model_config = ConfigDict(extra="forbid")

    policy_run_id: str
    policy_artifact: Literal["latest", "best"] = "best"
    engine_setting_raw_value: int = Field(default=50, ge=0, le=100)
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
    policy_run_id: str
    policy_artifact: Literal["latest", "best"] = "latest"


class UpdateDraftRequest(BaseModel):
    """Request body for updating one SQLite-backed draft."""

    model_config = ConfigDict(extra="forbid")

    name: str
    config: ManagedRunConfig
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
        recording_path: Path | None,
        target_kind: str | None,
        difficulty: str | None,
        cup_id: str | None,
        course_id: str | None,
        single_target: bool,
    ) -> Literal["started", "already_running"]: ...
