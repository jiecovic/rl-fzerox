# src/rl_fzerox/apps/run_manager/api.py
from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Literal, Protocol

from fastapi import FastAPI, HTTPException, Path, Query
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict
from starlette.requests import Request

from rl_fzerox.apps.run_manager.desktop import open_directory
from rl_fzerox.apps.run_manager.launch import ManagerRunLauncher
from rl_fzerox.apps.run_manager.tensorboard_metrics import (
    load_run_metric_samples_from_tensorboard,
)
from rl_fzerox.core.manager import (
    ManagedRun,
    ManagedRunConfig,
    ManagedRunDraft,
    ManagedRunEvent,
    ManagedRunMetricSample,
    ManagedRunTemplate,
    ManagerStore,
)
from rl_fzerox.core.manager.architecture import (
    policy_architecture_preview,
    run_manager_config_metadata,
)
from rl_fzerox.core.manager.errors import ManagerNameConflictError
from rl_fzerox.core.training.runs import RUN_LAYOUT
from rl_fzerox.core.training.session.callbacks.track_sampling import (
    TrackSamplingRuntimeState,
    load_track_sampling_runtime_state,
)


class CreateDraftRequest(BaseModel):
    """Request body for creating a SQLite-backed draft."""

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


def create_manager_api_app(
    store: ManagerStore,
    *,
    run_launcher: RunLauncher | None = None,
) -> FastAPI:
    """Create the local REST API app for the run manager."""

    app = FastAPI(title="F-Zero X Run Manager", version="0.1.0")
    launcher = run_launcher or ManagerRunLauncher(store)

    @app.exception_handler(HTTPException)
    async def handle_http_exception(_request: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})

    @app.exception_handler(RequestValidationError)
    async def handle_validation_exception(
        _request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        return JSONResponse(status_code=400, content={"error": jsonable_encoder(exc.errors())})

    @app.get("/api/health")
    def health() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/api/templates")
    def templates() -> dict[str, list[dict[str, object]]]:
        return {"templates": [_template_payload(item) for item in store.list_templates()]}

    @app.get("/api/drafts")
    def drafts() -> dict[str, list[dict[str, object]]]:
        return {"drafts": [_draft_payload(item) for item in store.list_drafts()]}

    @app.get("/api/runs")
    def runs() -> dict[str, list[dict[str, object]]]:
        visible_runs = store.list_visible_runs()
        recent_events = store.list_recent_run_events(
            tuple(run.id for run in visible_runs),
            limit_per_run=6,
        )
        return {
            "runs": [
                _run_payload(item, recent_events=recent_events.get(item.id, ()))
                for item in visible_runs
            ]
        }

    @app.put("/api/runs/{run_id}")
    def update_run(
        run_id: Annotated[str, Path(min_length=1)],
        request: UpdateRunRequest,
    ) -> dict[str, dict[str, object]]:
        name = request.name.strip()
        if not name:
            raise HTTPException(status_code=400, detail="run name is required")
        try:
            run = store.update_run_name(run_id=run_id, name=name)
        except ManagerNameConflictError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        if run is None:
            raise HTTPException(status_code=404, detail="run not found")
        recent_events = store.list_recent_run_events((run.id,), limit_per_run=6)
        return {"run": _run_payload(run, recent_events=recent_events.get(run.id, ()))}

    @app.get("/api/runs/{run_id}/metrics")
    def run_metrics(
        run_id: Annotated[str, Path(min_length=1)],
        mode: Literal["recent", "full"] = Query(default="recent"),
        limit: int = Query(default=240, ge=1, le=2_000),
    ) -> dict[str, list[dict[str, object]]]:
        run = store.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="run not found")
        samples = load_run_metric_samples_from_tensorboard(
            run,
            limit=None if mode == "full" else limit,
        )
        return {"samples": [_run_metric_payload(item) for item in samples]}

    @app.get("/api/runs/{run_id}/track-sampling")
    def run_track_sampling(
        run_id: Annotated[str, Path(min_length=1)],
    ) -> dict[str, object]:
        run = store.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="run not found")
        state = load_track_sampling_runtime_state(
            run.run_dir / RUN_LAYOUT.runtime_dirname / RUN_LAYOUT.track_sampling_state_filename,
        )
        return {"state": None if state is None else _track_sampling_state_payload(state)}

    @app.post("/api/runs", status_code=201)
    def launch_run(request: LaunchRunRequest) -> dict[str, dict[str, object]]:
        name = request.name.strip()
        if not name:
            raise HTTPException(status_code=400, detail="run name is required")
        _validate_source_fields(
            source_run_id=request.source_run_id,
            source_artifact=request.source_artifact,
        )
        if request.source_run_id is not None and request.draft_id is None:
            raise HTTPException(
                status_code=400,
                detail="fork launches must come from a persisted fork draft",
            )
        try:
            run = launcher.launch(
                name=name,
                config=request.config,
                draft_id=request.draft_id,
                source_run_id=request.source_run_id,
                source_artifact=request.source_artifact,
            )
        except ManagerNameConflictError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        recent_events = store.list_recent_run_events((run.id,), limit_per_run=6)
        return {"run": _run_payload(run, recent_events=recent_events.get(run.id, ()))}

    @app.post("/api/runs/{run_id}/fork", status_code=201)
    def fork_run(
        run_id: Annotated[str, Path(min_length=1)],
        request: ForkRunRequest,
    ) -> dict[str, dict[str, object]]:
        try:
            run = launcher.fork(
                run_id=run_id,
                artifact=request.artifact,
                name=request.name,
                config=request.config,
            )
        except ManagerNameConflictError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        recent_events = store.list_recent_run_events((run.id,), limit_per_run=6)
        return {"run": _run_payload(run, recent_events=recent_events.get(run.id, ()))}

    @app.post("/api/runs/{run_id}/pause")
    def pause_run(run_id: Annotated[str, Path(min_length=1)]) -> dict[str, dict[str, object]]:
        try:
            run = launcher.request_pause(run_id=run_id)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        recent_events = store.list_recent_run_events((run.id,), limit_per_run=6)
        return {"run": _run_payload(run, recent_events=recent_events.get(run.id, ()))}

    @app.post("/api/runs/{run_id}/stop")
    def stop_run(run_id: Annotated[str, Path(min_length=1)]) -> dict[str, dict[str, object]]:
        try:
            run = launcher.request_stop(run_id=run_id)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        recent_events = store.list_recent_run_events((run.id,), limit_per_run=6)
        return {"run": _run_payload(run, recent_events=recent_events.get(run.id, ()))}

    @app.post("/api/runs/{run_id}/resume")
    def resume_run(run_id: Annotated[str, Path(min_length=1)]) -> dict[str, dict[str, object]]:
        try:
            run = launcher.resume(run_id=run_id)
        except FileNotFoundError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        recent_events = store.list_recent_run_events((run.id,), limit_per_run=6)
        return {"run": _run_payload(run, recent_events=recent_events.get(run.id, ()))}

    @app.delete("/api/runs/{run_id}")
    def delete_run(run_id: Annotated[str, Path(min_length=1)]) -> dict[str, bool]:
        try:
            deleted = store.delete_run(run_id)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        if not deleted:
            raise HTTPException(status_code=404, detail="run not found")
        return {"deleted": True}

    @app.delete("/api/lineages/{lineage_id}")
    def delete_lineage(lineage_id: Annotated[str, Path(min_length=1)]) -> dict[str, bool]:
        try:
            deleted = store.delete_lineage(lineage_id)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        if not deleted:
            raise HTTPException(status_code=404, detail="lineage not found")
        return {"deleted": True}

    @app.post("/api/runs/{run_id}/open-dir")
    def open_run_dir(run_id: Annotated[str, Path(min_length=1)]) -> dict[str, bool]:
        run = store.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="run not found")
        try:
            open_directory(run.run_dir)
        except RuntimeError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return {"opened": True}

    @app.post("/api/runs/{run_id}/watch")
    def watch_run(
        run_id: Annotated[str, Path(min_length=1)],
        artifact: str = Query(default="latest"),
    ) -> dict[str, bool]:
        try:
            launcher.watch_artifact(run_id=run_id, artifact=artifact)
        except FileNotFoundError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return {"started": True}

    @app.get("/api/schema")
    def schema() -> dict[str, Mapping[str, object]]:
        return {"config": ManagedRunConfig.model_json_schema()}

    @app.get("/api/config-metadata")
    def config_metadata() -> dict[str, object]:
        return run_manager_config_metadata().model_dump(mode="json")

    @app.post("/api/policy-preview")
    def policy_preview(config: ManagedRunConfig) -> dict[str, object]:
        return policy_architecture_preview(config).model_dump(mode="json")

    @app.post("/api/drafts", status_code=201)
    def create_draft(request: CreateDraftRequest) -> dict[str, dict[str, object]]:
        name = request.name.strip()
        if not name:
            raise HTTPException(status_code=400, detail="draft name is required")
        _validate_source_fields(
            source_run_id=request.source_run_id,
            source_artifact=request.source_artifact,
        )
        try:
            draft = store.create_draft(
                name=name,
                config=request.config,
                source_run_id=request.source_run_id,
                source_artifact=request.source_artifact,
            )
        except ManagerNameConflictError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return {"draft": _draft_payload(draft)}

    @app.put("/api/drafts/{draft_id}")
    def update_draft(
        draft_id: Annotated[str, Path(min_length=1)],
        request: UpdateDraftRequest,
    ) -> dict[str, dict[str, object]]:
        name = request.name.strip()
        if not name:
            raise HTTPException(status_code=400, detail="draft name is required")
        _validate_source_fields(
            source_run_id=request.source_run_id,
            source_artifact=request.source_artifact,
        )
        try:
            draft = store.update_draft(
                draft_id=draft_id,
                name=name,
                config=request.config,
                source_run_id=request.source_run_id,
                source_artifact=request.source_artifact,
            )
        except ManagerNameConflictError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        if draft is None:
            raise HTTPException(status_code=404, detail="draft not found")
        return {"draft": _draft_payload(draft)}

    @app.delete("/api/drafts/{draft_id}")
    def delete_draft(
        draft_id: Annotated[str, Path(min_length=1)],
    ) -> dict[str, bool]:
        return {"deleted": store.delete_draft(draft_id)}

    return app


def _template_payload(template: ManagedRunTemplate) -> dict[str, object]:
    return {
        "id": template.id,
        "name": template.name,
        "created_at": template.created_at,
        "updated_at": template.updated_at,
        "config": template.config.model_dump(mode="json"),
    }


def _draft_payload(draft: ManagedRunDraft) -> dict[str, object]:
    return {
        "id": draft.id,
        "name": draft.name,
        "source_run_id": draft.source_run_id,
        "source_artifact": draft.source_artifact,
        "source_num_timesteps": draft.source_num_timesteps,
        "created_at": draft.created_at,
        "updated_at": draft.updated_at,
        "config": draft.config.model_dump(mode="json"),
    }


def _run_payload(
    run: ManagedRun,
    *,
    recent_events: tuple[ManagedRunEvent, ...] = (),
) -> dict[str, object]:
    return {
        "id": run.id,
        "name": run.name,
        "status": run.status,
        "created_at": run.created_at,
        "lineage_id": run.lineage_id,
        "lineage_step_offset": run.lineage_step_offset,
        "started_at": run.started_at,
        "stopped_at": run.stopped_at,
        "parent_run_id": run.parent_run_id,
        "source_run_id": run.source_run_id,
        "source_artifact": run.source_artifact,
        "source_num_timesteps": run.source_num_timesteps,
        "pending_command": run.pending_command,
        "runtime": None if run.runtime is None else {
            "total_timesteps": run.runtime.total_timesteps,
            "num_timesteps": run.runtime.num_timesteps,
            "progress_fraction": run.runtime.progress_fraction,
            "updated_at": run.runtime.updated_at,
            "fps": run.runtime.fps,
            "episode_reward_mean": run.runtime.episode_reward_mean,
            "episode_length_mean": run.runtime.episode_length_mean,
            "approx_kl": run.runtime.approx_kl,
            "entropy_loss": run.runtime.entropy_loss,
            "value_loss": run.runtime.value_loss,
            "policy_gradient_loss": run.runtime.policy_gradient_loss,
        },
        "recent_events": [
            {
                "created_at": event.created_at,
                "kind": event.kind,
                "message": event.message,
            }
            for event in recent_events
        ],
        "config": run.config.model_dump(mode="json"),
    }


def _run_metric_payload(sample: ManagedRunMetricSample) -> dict[str, object]:
    return {
        "run_id": sample.run_id,
        "created_at": sample.created_at,
        "total_timesteps": sample.total_timesteps,
        "num_timesteps": sample.num_timesteps,
        "lineage_num_timesteps": sample.lineage_num_timesteps,
        "progress_fraction": sample.progress_fraction,
        "metrics": sample.metrics,
        "fps": sample.fps,
        "episode_reward_mean": sample.episode_reward_mean,
        "episode_length_mean": sample.episode_length_mean,
        "approx_kl": sample.approx_kl,
        "entropy_loss": sample.entropy_loss,
        "value_loss": sample.value_loss,
        "policy_gradient_loss": sample.policy_gradient_loss,
    }


def _track_sampling_state_payload(state: TrackSamplingRuntimeState) -> dict[str, object]:
    total_weight = sum(entry.current_weight for entry in state.entries)
    total_episodes = sum(entry.episode_count for entry in state.entries)
    total_frames = sum(entry.completed_frames for entry in state.entries)
    return {
        "sampling_mode": state.sampling_mode,
        "action_repeat": state.action_repeat,
        "update_episodes": state.update_episodes,
        "update_count": state.update_count,
        "episodes_since_update": state.episodes_since_update,
        "entries": [
            {
                "track_id": entry.track_id,
                "course_key": entry.course_key,
                "label": entry.label,
                "current_weight": entry.current_weight,
                "current_probability": (
                    0.0 if total_weight <= 0.0 else entry.current_weight / total_weight
                ),
                "episode_count": entry.episode_count,
                "episode_share": (
                    0.0 if total_episodes <= 0 else entry.episode_count / total_episodes
                ),
                "completed_frames": entry.completed_frames,
                "completed_env_steps": (
                    0
                    if state.action_repeat <= 0
                    else entry.completed_frames // state.action_repeat
                ),
                "step_share": (
                    0.0 if total_frames <= 0 else entry.completed_frames / total_frames
                ),
            }
            for entry in state.entries
        ],
    }


def _validate_source_fields(
    *,
    source_run_id: str | None,
    source_artifact: Literal["latest", "best"] | None,
) -> None:
    if (source_run_id is None) != (source_artifact is None):
        raise HTTPException(
            status_code=400,
            detail="source_run_id and source_artifact must be set together",
        )
