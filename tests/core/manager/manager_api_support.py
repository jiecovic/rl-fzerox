# tests/core/manager/manager_api_support.py
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import httpx
from fastapi import FastAPI
from httpx._types import RequestFiles

from rl_fzerox.apps.run_manager.api import create_manager_api_app
from rl_fzerox.apps.run_manager.api.contracts import WatchRenderer
from rl_fzerox.core.manager import (
    ManagedRun,
    ManagedRunConfig,
    ManagerStore,
)
from rl_fzerox.core.training.session.callbacks.track_sampling import (
    TrackSamplingRuntimeEntry,
    TrackSamplingRuntimeState,
)


class _LauncherStub:
    def launch(
        self,
        *,
        name: str,
        config: ManagedRunConfig,
        draft_id: str | None,
        source_run_id: str | None,
        source_artifact: Literal["latest", "best"] | None,
        copy_alt_baselines: bool,
        engine_tuning_source_action: Literal["convert", "discard"],
    ) -> ManagedRun:
        del (
            name,
            config,
            draft_id,
            source_run_id,
            source_artifact,
            copy_alt_baselines,
            engine_tuning_source_action,
        )
        raise AssertionError("launch should not be called")

    def fork(
        self,
        *,
        run_id: str,
        artifact: Literal["latest", "best"],
        name: str | None,
        config: ManagedRunConfig | None,
        copy_alt_baselines: bool,
        engine_tuning_source_action: Literal["convert", "discard"],
    ) -> ManagedRun:
        del run_id, artifact, name, config, copy_alt_baselines, engine_tuning_source_action
        raise AssertionError("fork should not be called")

    def request_pause(self, *, run_id: str) -> ManagedRun:
        del run_id
        raise AssertionError("pause should not be called")

    def request_stop(self, *, run_id: str) -> ManagedRun:
        del run_id
        raise AssertionError("stop should not be called")

    def resume(self, *, run_id: str) -> ManagedRun:
        del run_id
        raise AssertionError("resume should not be called")

    def watch_artifact(
        self,
        *,
        run_id: str,
        artifact: str,
        device: Literal["cpu", "cuda"],
        renderer: WatchRenderer | None,
        deterministic_policy: bool,
    ) -> Literal["started", "already_running"]:
        del run_id, artifact, device, renderer, deterministic_policy
        raise AssertionError("watch should not be called")

    def start_career_mode(
        self,
        *,
        save_game_id: str,
        device: Literal["cpu", "cuda"],
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
    ) -> Literal["started", "already_running"]:
        del (
            save_game_id,
            device,
            renderer,
            attempt_seed,
            deterministic_policy,
            recording_enabled,
            recording_input_hud_enabled,
            recording_upscale_factor,
            recording_path,
            target_kind,
            difficulty,
            cup_id,
            course_id,
            single_target,
        )
        raise AssertionError("career mode runner should not be called")


class _ApiClient:
    def __init__(self, app: FastAPI) -> None:
        self._app = app

    async def request(
        self,
        method: str,
        url: str,
        *,
        content: str | bytes | None = None,
        files: RequestFiles | None = None,
        headers: Mapping[str, str] | None = None,
        json: object | None = None,
    ) -> httpx.Response:
        transport = httpx.ASGITransport(app=self._app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.request(
                method,
                url,
                content=content,
                files=files,
                headers=headers,
                json=json,
            )

    async def get(self, url: str) -> httpx.Response:
        return await self.request("GET", url)

    async def post(
        self,
        url: str,
        *,
        content: str | bytes | None = None,
        files: RequestFiles | None = None,
        headers: Mapping[str, str] | None = None,
        json: object | None = None,
    ) -> httpx.Response:
        return await self.request(
            "POST",
            url,
            content=content,
            files=files,
            headers=headers,
            json=json,
        )

    async def put(self, url: str, *, json: object | None = None) -> httpx.Response:
        return await self.request("PUT", url, json=json)

    async def delete(self, url: str, **kwargs: object) -> httpx.Response:
        del kwargs
        return await self.request("DELETE", url)


def _write_track_sampling_state(store: ManagerStore, run_id: str) -> None:
    store.upsert_run_track_sampling_state(
        run_id=run_id,
        state=TrackSamplingRuntimeState(
            sampling_mode="step_balanced",
            action_repeat=2,
            update_episodes=4,
            ema_alpha=0.5,
            max_weight_scale=5.0,
            adaptive_completion_weight=0.35,
            adaptive_target_completion=0.9,
            adaptive_min_confidence_episodes=24,
            adaptive_confidence_scale=4.0,
            update_count=3,
            episodes_since_update=1,
            entries=(
                TrackSamplingRuntimeEntry(
                    track_id="mute",
                    course_key="mute_city",
                    label="Mute City",
                    base_weight=1.0,
                    current_weight=1.5,
                    completed_frames=1200,
                    episode_count=3,
                    finished_episode_count=2,
                    success_sample_count=2,
                    ema_episode_frames=400.0,
                    ema_completion_fraction=None,
                ),
                TrackSamplingRuntimeEntry(
                    track_id="silence",
                    course_key="silence",
                    label="Silence",
                    base_weight=1.0,
                    current_weight=0.5,
                    completed_frames=800,
                    episode_count=1,
                    finished_episode_count=1,
                    success_sample_count=1,
                    ema_episode_frames=800.0,
                    ema_completion_fraction=None,
                ),
            ),
        ),
    )


def _write_policy_artifact(run_dir: Path, artifact: Literal["latest", "best"]) -> Path:
    policy_path = run_dir / "checkpoints" / artifact / "policy.zip"
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    policy_path.write_bytes(b"fake policy checkpoint")
    return policy_path


def _client(
    tmp_path: Path,
    *,
    launcher: _LauncherStub | None = None,
    store: ManagerStore | None = None,
) -> _ApiClient:
    resolved_store = store or ManagerStore(tmp_path / "manager" / "runs.db")
    return _ApiClient(
        create_manager_api_app(resolved_store, run_launcher=launcher or _LauncherStub())
    )
