# tests/core/manager/test_manager_api.py
from __future__ import annotations

import zipfile
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import rl_fzerox.apps.run_manager.api.handlers.metrics as manager_api_metrics
import rl_fzerox.apps.run_manager.api.handlers.save_games as manager_api_save_games
import rl_fzerox.core.manager.registry.runs.maintenance as run_maintenance
from rl_fzerox.apps.run_manager.api import create_manager_api_app
from rl_fzerox.apps.run_manager.api.contracts import WatchRenderer
from rl_fzerox.core.career_mode.progress import default_unlock_targets
from rl_fzerox.core.manager import (
    ManagedRun,
    ManagedRunConfig,
    ManagerStore,
    default_managed_run_config,
)
from rl_fzerox.core.manager.transfer import export_run_bundle
from rl_fzerox.core.training.session.callbacks.track_sampling import (
    TrackSamplingRuntimeEntry,
    TrackSamplingRuntimeState,
)

pytestmark = pytest.mark.anyio


class _LauncherStub:
    def launch(
        self,
        *,
        name: str,
        config: ManagedRunConfig,
        draft_id: str | None,
        source_run_id: str | None,
        source_artifact: Literal["latest", "best"] | None,
    ) -> ManagedRun:
        del name, config, draft_id, source_run_id, source_artifact
        raise AssertionError("launch should not be called")

    def fork(
        self,
        *,
        run_id: str,
        artifact: Literal["latest", "best"],
        name: str | None,
        config: ManagedRunConfig | None,
    ) -> ManagedRun:
        del run_id, artifact, name, config
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
    ) -> Literal["started", "already_running"]:
        del run_id, artifact, device, renderer
        raise AssertionError("watch should not be called")

    def start_career_mode(
        self,
        *,
        save_game_id: str,
        device: Literal["cpu", "cuda"],
        renderer: WatchRenderer | None,
        attempt_seed: int | None,
        deterministic_policy: bool,
        target_kind: str | None,
        difficulty: str | None,
        cup_id: str | None,
        course_id: str | None,
    ) -> Literal["started", "already_running"]:
        del (
            save_game_id,
            device,
            renderer,
            attempt_seed,
            deterministic_policy,
            target_kind,
            difficulty,
            cup_id,
            course_id,
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
        headers: Mapping[str, str] | None = None,
        json: object | None = None,
    ) -> httpx.Response:
        return await self.request("POST", url, content=content, headers=headers, json=json)

    async def put(self, url: str, *, json: object | None = None) -> httpx.Response:
        return await self.request("PUT", url, json=json)

    async def delete(self, url: str, **kwargs: object) -> httpx.Response:
        del kwargs
        return await self.request("DELETE", url)


async def test_manager_api_lists_default_template(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = await client.get("/api/templates")

    assert response.status_code == 200
    payload = response.json()
    assert payload["templates"][0]["id"] == "all_cups_recurrent_ppo"


async def test_manager_api_creates_draft(tmp_path: Path) -> None:
    client = _client(tmp_path)
    config = default_managed_run_config().model_dump(mode="json")

    response = await client.post("/api/drafts", json={"name": "Draft", "config": config})

    assert response.status_code == 201
    payload = response.json()
    assert payload["draft"]["name"] == "Draft"


async def test_manager_api_creates_save_game(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = await client.post("/api/save-games", json={"name": "Unlock Run"})

    assert response.status_code == 201
    payload = response.json()
    assert payload["save_game"]["name"] == "Unlock Run"
    assert "seed" not in payload["save_game"]
    assert payload["save_game"]["status"] == "created"
    assert payload["save_game"]["unlock_progress"]["completed_count"] == 0
    assert payload["save_game"]["unlock_progress"]["total_count"] == len(default_unlock_targets())
    assert payload["save_game"]["unlock_progress"]["next_target"]["difficulty"] == "novice"
    assert payload["save_game"]["unlock_progress"]["next_target"]["cup_id"] == "jack"
    assert payload["save_game"]["attempts"] == []
    assert payload["save_game"]["course_setups"] == []

    list_response = await client.get("/api/save-games")

    assert list_response.status_code == 200
    assert list_response.json()["save_games"] == [payload["save_game"]]


async def test_manager_api_resets_stale_unstarted_save_game_status(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    save_game = store.create_save_game(name="Unlock Run", save_games_root=tmp_path / "saves")
    updated = store.update_save_game_status(save_game_id=save_game.id, status="paused")
    assert updated is not None
    client = _client(tmp_path, store=store)

    response = await client.get("/api/save-games")

    assert response.status_code == 200
    payload = response.json()
    assert payload["save_games"][0]["id"] == save_game.id
    assert payload["save_games"][0]["status"] == "created"
    assert payload["save_games"][0]["attempts"] == []


async def test_manager_api_ignores_old_race_truncation_artifacts_for_start_state(
    tmp_path: Path,
) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    save_game = store.create_save_game(name="Unlock Run", save_games_root=tmp_path / "saves")
    attempt = store.start_save_attempt(
        save_game_id=save_game.id,
        target_kind="clear_gp_cup",
        difficulty="novice",
        cup_id="jack",
    )
    finished = store.finish_save_attempt(
        attempt_id=attempt.id,
        status="failed",
        finish_position=1,
        failure_reason="race truncated",
    )
    assert finished is not None
    updated = store.update_save_game_status(save_game_id=save_game.id, status="paused")
    assert updated is not None
    client = _client(tmp_path, store=store)

    response = await client.get("/api/save-games")

    assert response.status_code == 200
    payload = response.json()
    assert payload["save_games"][0]["id"] == save_game.id
    assert payload["save_games"][0]["status"] == "created"
    assert payload["save_games"][0]["attempts"][0]["failure_reason"] == "race truncated"


async def test_manager_api_upserts_save_course_setup(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    run = store.create_run(
        run_id="policy-run",
        name="Policy Run",
        config=default_managed_run_config(),
        managed_runs_root=tmp_path / "runs",
    )
    client = _client(tmp_path, store=store)
    create_response = await client.post("/api/save-games", json={"name": "Unlock Run"})
    save_game_id = create_response.json()["save_game"]["id"]

    response = await client.put(
        f"/api/save-games/{save_game_id}/course-setups",
        json={
            "policy_artifact": "best",
            "policy_run_id": run.id,
            "scope": "global",
        },
    )

    assert response.status_code == 200
    save_game = response.json()["save_game"]
    assignments = save_game["course_setups"]
    assert len(assignments) == 1
    assert assignments[0]["save_game_id"] == save_game_id
    assert assignments[0]["policy_run_id"] == "policy-run"
    assert assignments[0]["policy_artifact"] == "best"
    assert assignments[0]["scope"] == "global"


async def test_manager_api_starts_next_save_attempt(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    run = store.create_run(
        run_id="policy-run",
        name="Policy Run",
        config=default_managed_run_config(),
        managed_runs_root=tmp_path / "runs",
    )
    client = _client(tmp_path, store=store)
    create_response = await client.post("/api/save-games", json={"name": "Unlock Run"})
    save_game_id = create_response.json()["save_game"]["id"]
    await client.put(
        f"/api/save-games/{save_game_id}/course-setups",
        json={
            "policy_artifact": "best",
            "policy_run_id": run.id,
            "scope": "global",
        },
    )

    response = await client.post(f"/api/save-games/{save_game_id}/attempts/next")

    assert response.status_code == 200
    save_game = response.json()["save_game"]
    assert len(save_game["attempts"]) == 1
    assert save_game["attempts"][0]["save_game_id"] == save_game_id
    assert save_game["attempts"][0]["target_kind"] == "clear_gp_cup"
    assert save_game["attempts"][0]["difficulty"] == "novice"
    assert save_game["attempts"][0]["cup_id"] == "jack"
    assert save_game["attempts"][0]["policy_run_id"] == "policy-run"
    assert save_game["attempts"][0]["policy_artifact"] == "best"

    repeated_response = await client.post(f"/api/save-games/{save_game_id}/attempts/next")

    assert repeated_response.status_code == 400
    assert repeated_response.json()["error"] == "save game already has a running attempt"


async def test_manager_api_starts_career_mode_for_selected_target(tmp_path: Path) -> None:
    class _RecordingLauncher(_LauncherStub):
        request: dict[str, object] | None = None

        def start_career_mode(
            self,
            *,
            save_game_id: str,
            device: Literal["cpu", "cuda"],
            renderer: WatchRenderer | None,
            attempt_seed: int | None,
            deterministic_policy: bool,
            target_kind: str | None,
            difficulty: str | None,
            cup_id: str | None,
            course_id: str | None,
        ) -> Literal["started", "already_running"]:
            self.request = {
                "save_game_id": save_game_id,
                "device": device,
                "renderer": renderer,
                "attempt_seed": attempt_seed,
                "deterministic_policy": deterministic_policy,
                "target_kind": target_kind,
                "difficulty": difficulty,
                "cup_id": cup_id,
                "course_id": course_id,
            }
            return "started"

    launcher = _RecordingLauncher()
    client = _client(tmp_path, launcher=launcher)

    response = await client.post(
        "/api/save-games/save-001/runner",
        json={
            "device": "cpu",
            "renderer": "gliden64",
            "attempt_seed": 42,
            "policy_mode": "stochastic",
            "target_kind": "clear_gp_cup",
            "difficulty": "novice",
            "cup_id": "queen",
            "course_id": None,
        },
    )

    assert response.status_code == 200
    assert response.json() == {"status": "started"}
    assert launcher.request == {
        "save_game_id": "save-001",
        "device": "cpu",
        "renderer": "gliden64",
        "attempt_seed": 42,
        "deterministic_policy": False,
        "target_kind": "clear_gp_cup",
        "difficulty": "novice",
        "cup_id": "queen",
        "course_id": None,
    }


async def test_manager_api_returns_save_attempt_execution_context(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    run = store.create_run(
        run_id="policy-run",
        name="Policy Run",
        config=default_managed_run_config(),
        managed_runs_root=tmp_path / "runs",
    )
    policy_path = _write_policy_artifact(run.run_dir, "best")
    client = _client(tmp_path, store=store)
    create_response = await client.post("/api/save-games", json={"name": "Unlock Run"})
    save_game_id = create_response.json()["save_game"]["id"]
    await client.put(
        f"/api/save-games/{save_game_id}/course-setups",
        json={
            "policy_artifact": "best",
            "policy_run_id": run.id,
            "scope": "global",
        },
    )
    attempt_response = await client.post(f"/api/save-games/{save_game_id}/attempts/next")
    attempt = attempt_response.json()["save_game"]["attempts"][0]

    response = await client.get(f"/api/save-attempts/{attempt['id']}/execution-context")

    assert response.status_code == 200
    context = response.json()["execution_context"]
    assert context["attempt"]["id"] == attempt["id"]
    assert context["target"] == {
        "kind": "clear_gp_cup",
        "label": "Clear Novice Jack Cup",
        "difficulty": "novice",
        "cup_id": "jack",
        "course_id": None,
    }
    assert context["policy_run"]["id"] == "policy-run"
    assert context["policy_artifact"] == "best"
    assert context["policy_path"] == str(policy_path.resolve())

    plan_response = await client.get(f"/api/save-attempts/{attempt['id']}/execution-plan")

    assert plan_response.status_code == 200
    plan = plan_response.json()["execution_plan"]
    assert plan["attempt"]["id"] == attempt["id"]
    assert plan["policy"]["run_id"] == "policy-run"
    assert plan["policy"]["artifact"] == "best"
    assert plan["policy"]["path"] == str(policy_path.resolve())
    assert plan["race_setup"]["difficulty"] == "novice"
    assert plan["race_setup"]["cup_id"] == "jack"
    assert plan["race_setup"]["vehicle_id"] == "blue_falcon"


async def test_manager_api_rejects_duplicate_save_game_name(tmp_path: Path) -> None:
    client = _client(tmp_path)
    first_response = await client.post("/api/save-games", json={"name": "Unlock Run"})
    assert first_response.status_code == 201

    response = await client.post("/api/save-games", json={"name": "unlock run"})

    assert response.status_code == 409
    assert response.json()["error"] == "name already exists: unlock run"


async def test_manager_api_opens_save_game_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _client(tmp_path)
    create_response = await client.post("/api/save-games", json={"name": "Unlock Run"})
    payload = create_response.json()["save_game"]
    opened_paths: list[Path] = []

    monkeypatch.setattr(manager_api_save_games, "open_directory", opened_paths.append)

    response = await client.post(f"/api/save-games/{payload['id']}/open-dir")

    assert response.status_code == 200
    assert response.json() == {"opened": True}
    assert opened_paths == [Path(payload["save_path"]).parent]


async def test_manager_api_launches_run(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")

    class FakeLauncher(_LauncherStub):
        def launch(
            self,
            *,
            name: str,
            config: ManagedRunConfig,
            draft_id: str | None,
            source_run_id: str | None,
            source_artifact: Literal["latest", "best"] | None,
        ):
            del draft_id, source_artifact, source_run_id
            run = store.create_run(
                name=name,
                config=config,
                managed_runs_root=tmp_path / "runs",
            )
            launched = store.update_run_status(
                run_id=run.id,
                status="running",
                started_at="2026-05-03T12:00:00+00:00",
                stopped_at=None,
                message="worker launched",
            )
            if launched is None:
                raise RuntimeError("launch status update failed")
            return launched

    client = _ApiClient(create_manager_api_app(store, run_launcher=FakeLauncher()))
    config = default_managed_run_config().model_dump(mode="json")

    response = await client.post("/api/runs", json={"name": "Launch Me", "config": config})

    assert response.status_code == 201
    payload = response.json()
    assert payload["run"]["name"] == "Launch Me"
    assert payload["run"]["status"] == "running"


async def test_manager_api_watches_run_with_requested_policy_device(tmp_path: Path) -> None:
    class FakeLauncher(_LauncherStub):
        def __init__(self) -> None:
            self.calls: list[tuple[str, str, Literal["cpu", "cuda"], WatchRenderer | None]] = []

        def watch_artifact(
            self,
            *,
            run_id: str,
            artifact: str,
            device: Literal["cpu", "cuda"],
            renderer: WatchRenderer | None,
        ) -> Literal["started", "already_running"]:
            self.calls.append((run_id, artifact, device, renderer))
            return "started"

    launcher = FakeLauncher()
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    client = _ApiClient(create_manager_api_app(store, run_launcher=launcher))

    response = await client.post(
        "/api/runs/run-1/watch?artifact=best",
        json={"device": "cpu", "renderer": "angrylion"},
    )

    assert response.status_code == 200
    assert response.json() == {"status": "started"}
    assert launcher.calls == [("run-1", "best", "cpu", "angrylion")]


async def test_manager_api_watches_run_with_cuda_by_default(tmp_path: Path) -> None:
    class FakeLauncher(_LauncherStub):
        def __init__(self) -> None:
            self.device: Literal["cpu", "cuda"] | None = None
            self.renderer: WatchRenderer | None = None

        def watch_artifact(
            self,
            *,
            run_id: str,
            artifact: str,
            device: Literal["cpu", "cuda"],
            renderer: WatchRenderer | None,
        ) -> Literal["started", "already_running"]:
            del run_id, artifact
            self.device = device
            self.renderer = renderer
            return "already_running"

    launcher = FakeLauncher()
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    client = _ApiClient(create_manager_api_app(store, run_launcher=launcher))

    response = await client.post("/api/runs/run-1/watch")

    assert response.status_code == 200
    assert response.json() == {"status": "already_running"}
    assert launcher.device == "cuda"
    assert launcher.renderer is None


async def test_manager_api_launch_preserves_non_default_clip_range(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")

    class FakeLauncher(_LauncherStub):
        def launch(
            self,
            *,
            name: str,
            config: ManagedRunConfig,
            draft_id: str | None,
            source_run_id: str | None,
            source_artifact: Literal["latest", "best"] | None,
        ):
            del draft_id, source_artifact, source_run_id
            run = store.create_run(
                name=name,
                config=config,
                managed_runs_root=tmp_path / "runs",
            )
            launched = store.update_run_status(
                run_id=run.id,
                status="running",
                started_at="2026-05-03T12:00:00+00:00",
                stopped_at=None,
                message="worker launched",
            )
            if launched is None:
                raise RuntimeError("launch status update failed")
            return launched

    client = _ApiClient(create_manager_api_app(store, run_launcher=FakeLauncher()))
    config = default_managed_run_config().model_copy(
        update={"train": default_managed_run_config().train.model_copy(update={"clip_range": 0.17})}
    )

    response = await client.post(
        "/api/runs",
        json={"name": "Launch Clip", "config": config.model_dump(mode="json")},
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["run"]["config"]["train"]["clip_range"] == 0.17


async def test_manager_api_launches_unsaved_fork_run(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    seen_source: list[tuple[str | None, str | None, str | None]] = []
    source_run = store.create_run(
        run_id="run-parent",
        name="Parent Run",
        config=default_managed_run_config(),
        managed_runs_root=tmp_path / "runs",
    )

    class FakeLauncher(_LauncherStub):
        def launch(
            self,
            *,
            name: str,
            config: ManagedRunConfig,
            draft_id: str | None,
            source_run_id: str | None,
            source_artifact: Literal["latest", "best"] | None,
        ):
            seen_source.append((draft_id, source_run_id, source_artifact))
            run = store.create_run(
                name=name,
                config=config,
                managed_runs_root=tmp_path / "runs",
                source_run_id=source_run_id,
                source_artifact=source_artifact,
            )
            launched = store.update_run_status(
                run_id=run.id,
                status="running",
                started_at="2026-05-03T12:00:00+00:00",
                stopped_at=None,
                message="worker launched",
            )
            if launched is None:
                raise RuntimeError("launch status update failed")
            return launched

    client = _ApiClient(create_manager_api_app(store, run_launcher=FakeLauncher()))
    config = default_managed_run_config().model_dump(mode="json")

    response = await client.post(
        "/api/runs",
        json={
            "name": "Launch Fork",
            "config": config,
            "source_run_id": source_run.id,
            "source_artifact": "best",
        },
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["run"]["name"] == "Launch Fork"
    assert payload["run"]["source_run_id"] == source_run.id
    assert payload["run"]["source_artifact"] == "best"
    assert seen_source == [(None, source_run.id, "best")]


async def test_manager_api_launch_allows_same_name_as_source_draft(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")

    class FakeLauncher(_LauncherStub):
        def launch(
            self,
            *,
            name: str,
            config: ManagedRunConfig,
            draft_id: str | None,
            source_run_id: str | None,
            source_artifact: Literal["latest", "best"] | None,
        ):
            del source_artifact, source_run_id
            run = store.create_run(
                name=name,
                config=config,
                managed_runs_root=tmp_path / "runs",
                exclude_draft_id=draft_id,
            )
            launched = store.update_run_status(
                run_id=run.id,
                status="running",
                started_at="2026-05-03T12:00:00+00:00",
                stopped_at=None,
                message="worker launched",
            )
            if launched is None:
                raise RuntimeError("launch status update failed")
            return launched

    client = _ApiClient(create_manager_api_app(store, run_launcher=FakeLauncher()))
    config = default_managed_run_config().model_dump(mode="json")
    draft_response = await client.post("/api/drafts", json={"name": "Shared", "config": config})
    draft_id = draft_response.json()["draft"]["id"]

    response = await client.post(
        "/api/runs",
        json={"name": "Shared", "config": config, "draft_id": draft_id},
    )

    assert response.status_code == 201
    assert response.json()["run"]["name"] == "Shared"


async def test_manager_api_forks_run(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    parent = store.create_run(
        run_id="parent-run",
        name="Parent Run",
        config=default_managed_run_config(),
        explicit_run_dir=tmp_path / "runs" / "parent-run",
    )

    class FakeLauncher(_LauncherStub):
        def launch(
            self,
            *,
            name: str,
            config: ManagedRunConfig,
            draft_id: str | None,
            source_run_id: str | None,
            source_artifact: Literal["latest", "best"] | None,
        ):
            del config, draft_id, name, source_artifact, source_run_id
            raise AssertionError("launch should not be called")

        def fork(
            self,
            *,
            run_id: str,
            artifact: str,
            name: str | None,
            config: ManagedRunConfig | None,
        ):
            del config
            assert run_id == parent.id
            assert artifact == "best"
            child = store.create_run(
                run_id="child-run",
                name=name or "Parent Run best fork",
                config=default_managed_run_config(),
                explicit_run_dir=tmp_path / "runs" / "child-run",
                lineage_step_offset=816_040,
                parent_run_id=parent.id,
                source_run_id=parent.id,
                source_artifact=artifact,
                source_num_timesteps=816_040,
            )
            launched = store.update_run_status(
                run_id=child.id,
                status="running",
                started_at="2026-05-04T12:00:00+00:00",
                stopped_at=None,
                message="forked worker launched",
            )
            if launched is None:
                raise RuntimeError("fork status update failed")
            return launched

    client = _ApiClient(create_manager_api_app(store, run_launcher=FakeLauncher()))

    response = await client.post(f"/api/runs/{parent.id}/fork", json={"artifact": "best"})

    assert response.status_code == 201
    payload = response.json()
    assert payload["run"]["parent_run_id"] == parent.id
    assert payload["run"]["source_run_id"] == parent.id
    assert payload["run"]["source_artifact"] == "best"
    assert payload["run"]["source_num_timesteps"] == 816040


async def test_manager_api_reads_track_sampling_runtime_state(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    run = store.create_run(
        run_id="run-with-track-pool",
        name="Track Pool Run",
        config=default_managed_run_config(),
        explicit_run_dir=tmp_path / "runs" / "run-with-track-pool",
    )
    store.update_run_status(
        run_id=run.id,
        status="running",
        started_at="2026-05-04T12:00:00+00:00",
        stopped_at=None,
        message="worker launched",
    )
    _write_track_sampling_state(store, run.id)

    client = _client(tmp_path, store=store)

    response = await client.get(f"/api/runs/{run.id}/track-sampling")

    assert response.status_code == 200
    payload = response.json()["state"]
    assert payload["update_episodes"] == 4
    assert payload["entries"][0]["label"] == "Mute City"
    assert payload["entries"][0]["current_probability"] == pytest.approx(0.75)
    assert payload["entries"][0]["target_step_share"] == pytest.approx(0.5)
    assert payload["entries"][0]["completed_env_steps"] == 600
    assert payload["entries"][0]["finished_episode_count"] == 2
    assert payload["entries"][0]["success_sample_count"] == 2
    assert payload["entries"][0]["success_rate"] == pytest.approx(1.0)
    assert payload["entries"][0]["step_share"] == pytest.approx(0.6)


async def test_manager_api_reports_fixed_env_equal_sampling_share(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    run = store.create_run(
        run_id="run-with-fixed-env-track-pool",
        name="Fixed Env Track Pool Run",
        config=default_managed_run_config(),
        explicit_run_dir=tmp_path / "runs" / "run-with-fixed-env-track-pool",
    )
    store.upsert_run_track_sampling_state(
        run_id=run.id,
        state=TrackSamplingRuntimeState(
            sampling_mode="fixed_env",
            action_repeat=2,
            update_episodes=4,
            ema_alpha=0.5,
            max_weight_scale=5.0,
            adaptive_completion_weight=0.35,
            adaptive_target_completion=0.9,
            adaptive_min_confidence_episodes=24,
            adaptive_confidence_scale=4.0,
            update_count=0,
            episodes_since_update=0,
            entries=(
                TrackSamplingRuntimeEntry(
                    track_id="mute",
                    course_key="mute_city",
                    label="Mute City",
                    base_weight=1.0,
                    current_weight=1.0,
                    completed_frames=1200,
                    episode_count=3,
                    finished_episode_count=2,
                    success_sample_count=3,
                    ema_episode_frames=400.0,
                    ema_completion_fraction=0.8,
                ),
                TrackSamplingRuntimeEntry(
                    track_id="silence",
                    course_key="silence",
                    label="Silence",
                    base_weight=1.0,
                    current_weight=1.0,
                    completed_frames=800,
                    episode_count=1,
                    finished_episode_count=1,
                    success_sample_count=1,
                    ema_episode_frames=800.0,
                    ema_completion_fraction=1.0,
                ),
            ),
        ),
    )

    client = _client(tmp_path, store=store)

    response = await client.get(f"/api/runs/{run.id}/track-sampling")

    assert response.status_code == 200
    entries = response.json()["state"]["entries"]
    assert entries[0]["current_probability"] == pytest.approx(0.5)
    assert entries[1]["current_probability"] == pytest.approx(0.5)
    assert entries[0]["target_step_share"] == pytest.approx(0.0)
    assert entries[1]["target_step_share"] == pytest.approx(0.0)


def test_manager_api_live_track_sampling_sends_initial_snapshot(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    run = store.create_run(
        run_id="run-with-live-track-pool",
        name="Live Track Pool Run",
        config=default_managed_run_config(),
        explicit_run_dir=tmp_path / "runs" / "run-with-live-track-pool",
    )
    store.update_run_status(
        run_id=run.id,
        status="running",
        started_at="2026-05-04T12:00:00+00:00",
        stopped_at=None,
        message="worker launched",
    )
    _write_track_sampling_state(store, run.id)

    app = create_manager_api_app(store, run_launcher=_LauncherStub())

    with TestClient(app) as client:
        with client.websocket_connect(f"/api/runs/{run.id}/track-sampling/live") as websocket:
            payload = websocket.receive_json()

    assert payload["type"] == "track_sampling_snapshot"
    assert payload["state"]["entries"][0]["label"] == "Mute City"


async def test_manager_api_resets_track_sampling_state_for_stopped_run(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    run = store.create_run(
        run_id="run-reset-track-pool",
        name="Resettable Run",
        config=default_managed_run_config(),
        explicit_run_dir=tmp_path / "runs" / "run-reset-track-pool",
    )
    store.update_run_status(
        run_id=run.id,
        status="stopped",
        started_at="2026-05-04T12:00:00+00:00",
        stopped_at="2026-05-04T12:30:00+00:00",
        message="stopped for reset",
    )
    _write_track_sampling_state(store, run.id)

    client = _client(tmp_path, store=store)
    response = await client.post(f"/api/runs/{run.id}/track-sampling/reset")

    assert response.status_code == 200
    assert response.json() == {"reset": True}
    assert store.get_run_track_sampling_state(run.id) is None


async def test_manager_api_rejects_track_sampling_reset_while_running(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    run = store.create_run(
        run_id="run-active-track-pool",
        name="Active Run",
        config=default_managed_run_config(),
        explicit_run_dir=tmp_path / "runs" / "run-active-track-pool",
    )
    store.update_run_status(
        run_id=run.id,
        status="running",
        started_at="2026-05-04T12:00:00+00:00",
        stopped_at=None,
        message="worker launched",
    )

    client = _client(tmp_path, store=store)
    response = await client.post(f"/api/runs/{run.id}/track-sampling/reset")

    assert response.status_code == 400
    assert response.json()["error"] == "track-pool stats can only be reset while the run is stopped"


async def test_manager_api_deletes_lineage(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    run = store.create_run(
        run_id="root-run",
        name="Root Run",
        config=default_managed_run_config(),
        managed_runs_root=tmp_path / "runs",
    )
    store.create_run(
        run_id="leaf-run",
        name="Leaf Run",
        config=default_managed_run_config(),
        managed_runs_root=tmp_path / "runs",
        lineage_id=run.lineage_id,
        parent_run_id=run.id,
        source_run_id=run.id,
        source_artifact="latest",
        source_num_timesteps=111,
    )

    client = _client(tmp_path)
    response = await client.delete(f"/api/lineages/{run.lineage_id}")

    assert response.status_code == 200
    assert response.json() == {"deleted": True}


async def test_manager_api_updates_lineage_groups_and_tensorboard_view(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    run = store.create_run(
        run_id="root-run",
        name="Root Run",
        config=default_managed_run_config(),
        explicit_run_dir=tmp_path / "runs" / "root-run" / "root-run",
    )
    (run.run_dir / "tensorboard").mkdir(parents=True)
    store.update_run_status(run_id=run.id, status="stopped", message="stopped")

    client = _client(tmp_path, store=store)
    response = await client.put(
        f"/api/lineages/{run.lineage_id}/groups",
        json={"group_names": ["Old test runs", "Current ablations"]},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["lineage_groups"] == ["Current ablations", "Old test runs"]
    assert [view["slug"] for view in payload["tensorboard_views"]] == [
        "current-ablations",
        "old-test-runs",
    ]
    loaded = store.get_run(run.id)
    assert loaded is not None
    assert loaded.lineage_groups == ("Current ablations", "Old test runs")


async def test_manager_api_metrics_full_mode_disables_recent_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    run = store.create_run(
        run_id="run-001",
        name="Metrics Run",
        config=default_managed_run_config(),
        explicit_run_dir=tmp_path / "runs" / "run-001",
    )
    seen_limits: list[int | None] = []

    def fake_loader(
        managed_run: ManagedRun,
        *,
        limit: int | None,
    ) -> tuple[object, ...]:
        assert managed_run.id == run.id
        seen_limits.append(limit)
        return ()

    monkeypatch.setattr(
        manager_api_metrics,
        "load_run_metric_samples_from_tensorboard",
        fake_loader,
    )

    class FakeLauncher(_LauncherStub):
        def launch(
            self,
            *,
            name: str,
            config: ManagedRunConfig,
            draft_id: str | None,
            source_run_id: str | None,
            source_artifact: Literal["latest", "best"] | None,
        ) -> ManagedRun:
            del name, config, draft_id, source_run_id, source_artifact
            raise AssertionError("launch should not be called")

    client = _ApiClient(create_manager_api_app(store, run_launcher=FakeLauncher()))

    recent_response = await client.get(f"/api/runs/{run.id}/metrics")
    full_response = await client.get(f"/api/runs/{run.id}/metrics?mode=full")

    assert recent_response.status_code == 200
    assert full_response.status_code == 200
    assert seen_limits == [240, None]


async def test_manager_api_updates_draft(tmp_path: Path) -> None:
    client = _client(tmp_path)
    create_response = await client.post(
        "/api/drafts",
        json={"name": "Draft", "config": default_managed_run_config().model_dump(mode="json")},
    )
    draft_id = create_response.json()["draft"]["id"]
    updated_config = default_managed_run_config().model_dump(mode="json")
    updated_config["seed"] = 999

    response = await client.put(
        f"/api/drafts/{draft_id}",
        json={"name": "Updated draft", "config": updated_config},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["draft"]["name"] == "Updated draft"
    assert payload["draft"]["config"]["seed"] == 999


async def test_manager_api_rejects_duplicate_draft_name(tmp_path: Path) -> None:
    client = _client(tmp_path)
    config = default_managed_run_config().model_dump(mode="json")
    first_response = await client.post("/api/drafts", json={"name": "Draft", "config": config})

    response = await client.post("/api/drafts", json={"name": "draft", "config": config})

    assert first_response.status_code == 201
    assert response.status_code == 409
    assert response.json()["error"] == "name already exists: draft"


async def test_manager_api_rejects_renaming_draft_to_existing_name(tmp_path: Path) -> None:
    client = _client(tmp_path)
    config = default_managed_run_config().model_dump(mode="json")
    first_draft_id = (
        await client.post("/api/drafts", json={"name": "Alpha", "config": config})
    ).json()["draft"]["id"]
    await client.post("/api/drafts", json={"name": "Beta", "config": config})

    response = await client.put(
        f"/api/drafts/{first_draft_id}",
        json={"name": "beta", "config": config},
    )

    assert response.status_code == 409
    assert response.json()["error"] == "name already exists: beta"


async def test_manager_api_allows_draft_name_matching_existing_run(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    store.create_run(
        name="Shared Name",
        config=default_managed_run_config(),
        managed_runs_root=tmp_path / "runs",
    )
    client = _ApiClient(create_manager_api_app(store))
    config = default_managed_run_config().model_dump(mode="json")

    response = await client.post("/api/drafts", json={"name": "Shared Name", "config": config})

    assert response.status_code == 201
    assert response.json()["draft"]["name"] == "Shared Name"


async def test_manager_api_rejects_invalid_json(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = await client.post(
        "/api/drafts",
        content="{",
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 400
    assert "error" in response.json()


async def test_manager_api_rejects_missing_draft_delete(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = await client.delete("/api/drafts/missing-draft")

    assert response.status_code == 404
    assert response.json()["error"] == "draft not found"


async def test_manager_api_hides_unstarted_run_records(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    store.create_run(
        name="Created Only",
        config=default_managed_run_config(),
        managed_runs_root=tmp_path / "runs",
    )
    client = _ApiClient(create_manager_api_app(store))

    response = await client.get("/api/runs")

    assert response.status_code == 200
    assert response.json() == {"runs": []}


async def test_manager_api_lists_run_summaries_and_fetches_run_details(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    config = default_managed_run_config().model_copy(update={"seed": 1234})
    run = store.create_run(
        name="Visible Run",
        config=config,
        managed_runs_root=tmp_path / "runs",
    )
    updated = store.update_run_status(
        run_id=run.id,
        status="stopped",
        message="run stopped",
        stopped_at="2026-05-03T12:00:00+00:00",
    )
    if updated is None:
        raise RuntimeError("status update failed")
    client = _ApiClient(create_manager_api_app(store))

    list_response = await client.get("/api/runs")

    assert list_response.status_code == 200
    summary = list_response.json()["runs"][0]
    assert summary["name"] == "Visible Run"
    assert summary["config_hash"] == run.config_hash
    assert summary["action_repeat"] == config.action.action_repeat
    assert "config" not in summary

    detail_response = await client.get(f"/api/runs/{run.id}")

    assert detail_response.status_code == 200
    detail = detail_response.json()["run"]
    assert detail["config"]["seed"] == 1234
    assert detail["config_hash"] == run.config_hash


def test_manager_api_live_runs_sends_initial_snapshot(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    run = store.create_run(
        name="Visible Live Run",
        config=default_managed_run_config(),
        managed_runs_root=tmp_path / "runs",
    )
    store.update_run_status(run_id=run.id, status="stopped", message="run stopped")

    app = create_manager_api_app(store, run_launcher=_LauncherStub())

    with TestClient(app) as client:
        with client.websocket_connect("/api/runs/live") as websocket:
            payload = websocket.receive_json()

    assert payload["type"] == "runs_snapshot"
    assert payload["runs"][0]["id"] == run.id


def test_manager_api_exports_run_bundle(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    run = store.create_run(
        run_id="run-a",
        name="Export Run",
        config=default_managed_run_config(),
        explicit_run_dir=tmp_path / "runs" / "run-a" / "run-a",
    )
    run.run_dir.mkdir(parents=True)
    (run.run_dir / "train_config.yaml").write_text("run_name: run-a\n", encoding="utf-8")
    store.update_run_status(run_id=run.id, status="stopped", message="stopped")

    with TestClient(create_manager_api_app(store, run_launcher=_LauncherStub())) as client:
        response = client.get(f"/api/runs/{run.id}/export")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"
    bundle_path = tmp_path / "export.zip"
    bundle_path.write_bytes(response.content)
    with zipfile.ZipFile(bundle_path) as archive:
        assert "run_export.json" in archive.namelist()
        assert "run/train_config.yaml" in archive.namelist()


def test_manager_api_imports_run_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_store = ManagerStore(tmp_path / "source" / "manager" / "runs.db")
    source_run_dir = tmp_path / "source" / "runs" / "run-a" / "run-a"
    source_run = source_store.create_run(
        run_id="run-a",
        name="Import Run",
        config=default_managed_run_config(),
        explicit_run_dir=source_run_dir,
    )
    source_run_dir.mkdir(parents=True)
    (source_run_dir / "train_config.yaml").write_text(
        f"explicit_run_dir: {source_run_dir}\n",
        encoding="utf-8",
    )
    source_store.update_run_status(run_id=source_run.id, status="stopped", message="stopped")
    bundle_path = export_run_bundle(
        store=source_store,
        run_id=source_run.id,
        output_path=tmp_path / "run-a.zip",
    )
    target_store = ManagerStore(tmp_path / "target" / "manager" / "runs.db")

    def target_runs_root(*, output_root: Path | None = None) -> Path:
        return tmp_path / "target" / "runs" if output_root is None else output_root

    monkeypatch.setattr(target_store, "manager_runs_root", target_runs_root)

    with TestClient(create_manager_api_app(target_store, run_launcher=_LauncherStub())) as client:
        with bundle_path.open("rb") as bundle:
            response = client.post(
                "/api/run-imports",
                files={"bundle": ("run-a.zip", bundle, "application/zip")},
            )

    assert response.status_code == 201
    payload = response.json()
    imported_run = target_store.get_run("run-a")
    assert imported_run is not None
    assert payload["run"]["id"] == "run-a"
    assert imported_run.run_dir == tmp_path / "target" / "runs" / "run-a" / "run-a"
    imported_manifest = (imported_run.run_dir / "train_config.yaml").read_text(encoding="utf-8")
    assert str(source_run_dir) not in imported_manifest
    assert str(imported_run.run_dir) in imported_manifest


async def test_manager_api_exposes_worker_heartbeat_separately_from_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    run = store.create_run(
        name="Running Run",
        config=default_managed_run_config(),
        managed_runs_root=tmp_path / "runs",
    )
    now = datetime.now(UTC)
    started_at = now.isoformat(timespec="seconds")
    heartbeat_at = (now + timedelta(seconds=3)).isoformat(timespec="seconds")
    runtime_updated_at = (now + timedelta(seconds=1)).isoformat(timespec="seconds")
    launched = store.update_run_status(
        run_id=run.id,
        status="running",
        started_at=started_at,
        stopped_at=None,
        message="worker launched",
    )
    if launched is None:
        raise RuntimeError("launch status update failed")
    assert store.register_run_worker(
        run_id=run.id,
        launch_token="token-1",
        pid=12345,
        launched_at=started_at,
    )
    assert store.heartbeat_run_worker(
        run_id=run.id,
        launch_token="token-1",
        heartbeat_at=heartbeat_at,
    )
    store.upsert_run_runtime(
        run_id=run.id,
        total_timesteps=1000,
        num_timesteps=500,
        progress_fraction=0.5,
        updated_at=runtime_updated_at,
    )
    monkeypatch.setattr(run_maintenance, "pid_exists", lambda pid: True)
    client = _ApiClient(create_manager_api_app(store))

    response = await client.get("/api/runs")

    assert response.status_code == 200
    payload = response.json()
    assert payload["runs"][0]["worker_heartbeat_at"] == heartbeat_at
    assert payload["runs"][0]["runtime"]["updated_at"] == runtime_updated_at


async def test_manager_api_exposes_config_metadata(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = await client.get("/api/config-metadata")

    assert response.status_code == 200
    payload = response.json()
    preset_values = {preset["value"] for preset in payload["observation_presets"]}
    assert preset_values == {"crop_72x96", "crop_84x84"}
    preset_labels = {preset["value"]: preset["label"] for preset in payload["observation_presets"]}
    assert preset_labels["crop_72x96"] == "72 x 96 IMPALA"
    assert preset_labels["crop_84x84"] == "84 x 84 DQN/Atari"
    assert "gliden64" in {source["renderer"] for source in payload["observation_source_geometries"]}
    assert "nature" in {profile["value"] for profile in payload["conv_profiles"]}
    assert "impala_small" in {profile["value"] for profile in payload["conv_profiles"]}
    assert "impala_large" in {profile["value"] for profile in payload["conv_profiles"]}
    assert "custom" in {profile["value"] for profile in payload["conv_profiles"]}
    assert "time_attack" in {mode["value"] for mode in payload["race_modes"]}
    assert "master" in {mode["value"] for mode in payload["gp_difficulties"]}
    assert "step_balanced" in {mode["value"] for mode in payload["track_sampling_modes"]}
    assert "adaptive_step_balanced" in {mode["value"] for mode in payload["track_sampling_modes"]}
    assert "deficit_budget" in {mode["value"] for mode in payload["track_sampling_modes"]}
    assert "jack" in {cup["id"] for cup in payload["track_cups"]}
    assert "mute_city" in {course["id"] for course in payload["built_in_courses"]}
    assert "blue_falcon" in {vehicle["id"] for vehicle in payload["vehicles"]}
    blue_falcon = next(vehicle for vehicle in payload["vehicles"] if vehicle["id"] == "blue_falcon")
    assert blue_falcon["menu_row"] == 0
    assert blue_falcon["menu_column"] == 0
    red_gazelle = next(vehicle for vehicle in payload["vehicles"] if vehicle["id"] == "red_gazelle")
    assert red_gazelle["menu_row"] == 0
    assert red_gazelle["menu_column"] == 5
    assert "balanced" in {preset["id"] for preset in payload["engine_setting_presets"]}
    assert "continuous" in {mode["value"] for mode in payload["steering_modes"]}
    assert "on_off" in {mode["value"] for mode in payload["drive_modes"]}
    lean_output_modes = {mode["value"] for mode in payload["lean_output_modes"]}
    assert "four_way_categorical" in lean_output_modes
    assert "independent_buttons" in lean_output_modes
    lean_modes = {mode["value"] for mode in payload["lean_modes"]}
    assert "release_cooldown" in lean_modes
    assert "raw" in lean_modes


async def test_manager_api_accepts_frontend_action_layouts_even_if_runtime_support_lags(
    tmp_path: Path,
) -> None:
    client = _client(tmp_path)
    config = default_managed_run_config().model_dump(mode="json")
    config["action"]["steering_mode"] = "discrete"
    config["action"]["drive_mode"] = "pwm"
    config["action"]["include_air_brake"] = False
    config["action"]["include_pitch"] = False

    response = await client.post("/api/drafts", json={"name": "Draft", "config": config})

    assert response.status_code == 201


async def test_manager_api_previews_policy_architecture(tmp_path: Path) -> None:
    client = _client(tmp_path)
    config = default_managed_run_config().model_dump(mode="json")

    response = await client.post("/api/policy-preview", json=config)

    assert response.status_code == 200
    payload = response.json()
    assert payload["image_shape"] == {"height": 84, "width": 84, "channels": 6}
    assert payload["total_params"] > 0
    assert payload["continuous_action_dims"] == 1
    assert payload["discrete_action_logits"] == 14
    assert payload["architecture_lanes"][0]["label"] == "Image branch"
    cnn_node = next(
        node for node in payload["architecture_lanes"][0]["nodes"] if node["id"] == "cnn"
    )
    assert cnn_node["params"] > 0
    fusion_nodes = payload["architecture_lanes"][2]["nodes"]
    assert {node["id"] for node in fusion_nodes} >= {
        "action_net",
        "policy_head",
        "value_head",
        "value_net",
    }
    action_branch_names = {branch["name"] for branch in payload["action_branches"]}
    assert action_branch_names == {
        "steer",
        "throttle",
        "air_brake",
        "boost",
        "lean",
        "pitch",
    }


async def test_manager_api_previews_raw_state_fusion_without_state_mlp(tmp_path: Path) -> None:
    client = _client(tmp_path)
    config = default_managed_run_config().model_dump(mode="json")
    config["policy"]["state_net_arch"] = []

    response = await client.post("/api/policy-preview", json=config)

    assert response.status_code == 200
    payload = response.json()
    assert payload["state_features_dim"] == payload["state_dim"]
    state_nodes = payload["architecture_lanes"][1]["nodes"]
    state_mlp = next(node for node in state_nodes if node["id"] == "state_mlp")
    assert state_mlp["tone"] == "muted"


async def test_manager_api_previews_custom_cnn_profile(tmp_path: Path) -> None:
    client = _client(tmp_path)
    config = default_managed_run_config().model_dump(mode="json")
    config["policy"]["conv_profile"] = "custom"
    config["policy"]["custom_conv_layers"] = [
        {"kind": "conv", "out_channels": 16, "kernel_size": 3, "stride": 2, "padding": 1},
        {
            "kind": "residual_post",
            "out_channels": 16,
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
        },
        {"kind": "maxpool", "out_channels": 16, "kernel_size": 2, "stride": 2, "padding": 0},
        {"kind": "avgpool", "out_channels": 16, "kernel_size": 3, "stride": 1, "padding": 1},
        {"kind": "conv", "out_channels": 32, "kernel_size": 4, "stride": 2, "padding": 0},
        {"kind": "conv", "out_channels": 48, "kernel_size": 3, "stride": 1, "padding": 1},
    ]

    response = await client.post("/api/policy-preview", json=config)

    assert response.status_code == 200
    payload = response.json()
    assert [layer["name"] for layer in payload["conv_layers"]] == [
        "conv1",
        "res2",
        "pool3",
        "avgpool4",
        "conv5",
        "conv6",
    ]
    assert [layer["kind"] for layer in payload["conv_layers"]] == [
        "conv",
        "residual_post",
        "maxpool",
        "avgpool",
        "conv",
        "conv",
    ]
    assert [layer["out_channels"] for layer in payload["conv_layers"]] == [16, 16, 16, 16, 32, 48]
    assert [layer["padding"] for layer in payload["conv_layers"]] == [1, 1, 0, 1, 0, 1]
    assert payload["conv_layers"][0]["output_height"] == 42
    assert payload["conv_layers"][0]["output_width"] == 42
    assert payload["conv_layers"][1]["output_height"] == 42
    assert payload["conv_layers"][1]["output_width"] == 42
    assert payload["conv_layers"][2]["output_height"] == 21
    assert payload["conv_layers"][2]["output_width"] == 21
    assert payload["conv_layers"][3]["output_height"] == 21
    assert payload["conv_layers"][3]["output_width"] == 21


async def test_manager_api_previews_impala_small_cnn_profile(tmp_path: Path) -> None:
    client = _client(tmp_path)
    config = default_managed_run_config().model_dump(mode="json")
    config["observation"]["resolution"] = {"mode": "preset", "preset": "crop_72x96"}
    config["policy"]["conv_profile"] = "impala_small"

    response = await client.post("/api/policy-preview", json=config)

    assert response.status_code == 200
    payload = response.json()
    layer_shapes = [
        (layer["out_channels"], layer["output_height"], layer["output_width"])
        for layer in payload["conv_layers"]
    ]
    assert layer_shapes == [(16, 17, 23), (32, 7, 10)]
    assert payload["image_features_dim"] == 2_240


async def test_manager_api_previews_impala_large_cnn_profile(tmp_path: Path) -> None:
    client = _client(tmp_path)
    config = default_managed_run_config().model_dump(mode="json")
    config["observation"]["resolution"] = {"mode": "preset", "preset": "crop_72x96"}
    config["policy"]["conv_profile"] = "impala_large"

    response = await client.post("/api/policy-preview", json=config)

    assert response.status_code == 200
    payload = response.json()
    assert payload["image_shape"] == {"height": 72, "width": 96, "channels": 6}
    assert [layer["kind"] for layer in payload["conv_layers"]] == [
        "conv",
        "maxpool",
        "residual_pre",
        "residual_pre",
        "conv",
        "maxpool",
        "residual_pre",
        "residual_pre",
        "conv",
        "maxpool",
        "residual_pre",
        "residual_pre",
        "activation",
    ]
    assert [layer["post_activation"] for layer in payload["conv_layers"][:2]] == [False, True]
    layer_shapes = [
        (layer["out_channels"], layer["output_height"], layer["output_width"])
        for layer in payload["conv_layers"]
    ]
    pixel_drops = [
        (layer["dropped_height"], layer["dropped_width"]) for layer in payload["conv_layers"]
    ]
    assert layer_shapes == [
        (16, 72, 96),
        (16, 36, 48),
        (16, 36, 48),
        (16, 36, 48),
        (32, 36, 48),
        (32, 18, 24),
        (32, 18, 24),
        (32, 18, 24),
        (32, 18, 24),
        (32, 9, 12),
        (32, 9, 12),
        (32, 9, 12),
        (32, 9, 12),
    ]
    assert pixel_drops == [(0, 0)] * 13
    assert payload["image_features_dim"] == 3_456


async def test_manager_api_preview_keeps_masked_branch_logits_but_marks_status(
    tmp_path: Path,
) -> None:
    client = _client(tmp_path)
    config = default_managed_run_config().model_dump(mode="json")
    config["action"]["enable_boost"] = False

    response = await client.post("/api/policy-preview", json=config)

    assert response.status_code == 200
    payload = response.json()
    assert payload["discrete_action_logits"] == 14
    boost_branch = next(
        branch for branch in payload["action_branches"] if branch["name"] == "boost"
    )
    assert boost_branch["enabled"] is False
    assert boost_branch["mask_label"] == "masked idle"


async def test_manager_api_preview_keeps_gas_head_but_marks_forced_full(
    tmp_path: Path,
) -> None:
    client = _client(tmp_path)
    config = default_managed_run_config().model_dump(mode="json")
    config["action"]["force_full_throttle"] = True

    response = await client.post("/api/policy-preview", json=config)

    assert response.status_code == 200
    payload = response.json()
    assert payload["continuous_action_dims"] == 1
    assert payload["discrete_action_logits"] == 14
    throttle_branch = next(
        branch for branch in payload["action_branches"] if branch["name"] == "throttle"
    )
    assert throttle_branch["enabled"] is False
    assert throttle_branch["mask_label"] == "forced engaged"


async def test_manager_api_preview_removes_logits_when_branch_excluded(tmp_path: Path) -> None:
    client = _client(tmp_path)
    config = default_managed_run_config().model_dump(mode="json")
    config["action"]["include_boost"] = False

    response = await client.post("/api/policy-preview", json=config)

    assert response.status_code == 200
    payload = response.json()
    assert payload["discrete_action_logits"] == 12
    branch_names = {branch["name"] for branch in payload["action_branches"]}
    assert "boost" not in branch_names


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
