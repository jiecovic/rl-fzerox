# src/rl_fzerox/apps/run_manager/api/contracts.py
from __future__ import annotations

from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from rl_fzerox.core.manager import CourseSetupScope, ManagedRun, ManagedRunConfig

WatchDevice = Literal["cpu", "cuda"]
WatchRenderer = Literal["angrylion", "gliden64"]
PolicyPlaybackMode = Literal["deterministic", "stochastic"]


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


class UpdateSaveGameRequest(BaseModel):
    """Request body for renaming one manager-owned save game."""

    model_config = ConfigDict(extra="forbid")

    name: str


class UpsertSaveCourseSetupRequest(BaseModel):
    """Request body for assigning a policy to one save-game scope."""

    model_config = ConfigDict(extra="forbid")

    scope: CourseSetupScope
    policy_run_id: str
    policy_artifact: Literal["latest", "best"] = "best"
    vehicle_id: str
    engine_setting_raw_value: int = Field(ge=0, le=100)
    difficulty: str | None = None
    cup_id: str | None = None
    course_id: str | None = None


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


class ForkRunRequest(BaseModel):
    """Request body for forking one managed training run."""

    model_config = ConfigDict(extra="forbid")

    artifact: Literal["latest", "best"]
    name: str | None = None
    config: ManagedRunConfig | None = None


class RunLauncher(Protocol):
    def launch(
        self,
        *,
        name: str,
        config: ManagedRunConfig,
        draft_id: str | None,
        source_run_id: str | None,
        source_artifact: Literal["latest", "best"] | None,
    ) -> ManagedRun: ...

    def fork(
        self,
        *,
        run_id: str,
        artifact: Literal["latest", "best"],
        name: str | None,
        config: ManagedRunConfig | None,
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
    ) -> Literal["started", "already_running"]: ...

    def start_career_mode(
        self,
        *,
        save_game_id: str,
        device: WatchDevice,
        renderer: WatchRenderer | None,
        attempt_seed: int | None,
        deterministic_policy: bool,
    ) -> Literal["started", "already_running"]: ...
