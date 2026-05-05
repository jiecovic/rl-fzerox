# src/rl_fzerox/apps/run_manager/api/contracts.py
from __future__ import annotations

from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict

from rl_fzerox.core.manager import ManagedRun, ManagedRunConfig


class CreateDraftRequest(BaseModel):
    """Request body for creating one SQLite-backed draft."""

    model_config = ConfigDict(extra="forbid")

    name: str
    config: ManagedRunConfig
    source_run_id: str | None = None
    source_artifact: Literal["latest", "best"] | None = None


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

    def watch_artifact(self, *, run_id: str, artifact: str) -> None: ...
