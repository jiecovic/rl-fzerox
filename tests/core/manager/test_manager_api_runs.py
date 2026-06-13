# tests/core/manager/test_manager_api_runs.py
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

import pytest

import rl_fzerox.core.manager.registry.runs.maintenance as run_maintenance
from rl_fzerox.apps.run_manager.api import create_manager_api_app
from rl_fzerox.apps.run_manager.api.contracts import WatchRenderer
from rl_fzerox.core.manager import (
    ManagedRunConfig,
    ManagerStore,
    default_managed_run_config,
)
from rl_fzerox.core.training.session.callbacks.track_sampling import (
    TrackSamplingAltBaseline,
    TrackSamplingRuntimeEntry,
    TrackSamplingRuntimeState,
)
from tests.core.manager.manager_api_support import (
    _ApiClient,
    _client,
    _LauncherStub,
    _write_track_sampling_state,
)

pytestmark = pytest.mark.anyio


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
            copy_alt_baselines: bool,
        ):
            del copy_alt_baselines, draft_id, source_artifact, source_run_id
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
            self.calls: list[
                tuple[str, str, Literal["cpu", "cuda"], WatchRenderer | None, bool]
            ] = []

        def watch_artifact(
            self,
            *,
            run_id: str,
            artifact: str,
            device: Literal["cpu", "cuda"],
            renderer: WatchRenderer | None,
            deterministic_policy: bool,
        ) -> Literal["started", "already_running"]:
            self.calls.append((run_id, artifact, device, renderer, deterministic_policy))
            return "started"

    launcher = FakeLauncher()
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    client = _ApiClient(create_manager_api_app(store, run_launcher=launcher))

    response = await client.post(
        "/api/runs/run-1/watch?artifact=best",
        json={"device": "cpu", "renderer": "angrylion", "policy_mode": "stochastic"},
    )

    assert response.status_code == 200
    assert response.json() == {"status": "started"}
    assert launcher.calls == [("run-1", "best", "cpu", "angrylion", False)]


async def test_manager_api_watches_run_with_cuda_by_default(tmp_path: Path) -> None:
    class FakeLauncher(_LauncherStub):
        def __init__(self) -> None:
            self.device: Literal["cpu", "cuda"] | None = None
            self.renderer: WatchRenderer | None = None
            self.deterministic_policy: bool | None = None

        def watch_artifact(
            self,
            *,
            run_id: str,
            artifact: str,
            device: Literal["cpu", "cuda"],
            renderer: WatchRenderer | None,
            deterministic_policy: bool,
        ) -> Literal["started", "already_running"]:
            del run_id, artifact
            self.device = device
            self.renderer = renderer
            self.deterministic_policy = deterministic_policy
            return "already_running"

    launcher = FakeLauncher()
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    client = _ApiClient(create_manager_api_app(store, run_launcher=launcher))

    response = await client.post("/api/runs/run-1/watch")

    assert response.status_code == 200
    assert response.json() == {"status": "already_running"}
    assert launcher.device == "cuda"
    assert launcher.renderer is None
    assert launcher.deterministic_policy is True


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
            copy_alt_baselines: bool,
        ):
            del copy_alt_baselines, draft_id, source_artifact, source_run_id
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
    seen_source: list[tuple[str | None, str | None, str | None, bool]] = []
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
            copy_alt_baselines: bool,
        ):
            seen_source.append((draft_id, source_run_id, source_artifact, copy_alt_baselines))
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
            "copy_alt_baselines": False,
        },
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["run"]["name"] == "Launch Fork"
    assert payload["run"]["source_run_id"] == source_run.id
    assert payload["run"]["source_artifact"] == "best"
    assert seen_source == [(None, source_run.id, "best", False)]


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
            copy_alt_baselines: bool,
        ):
            del copy_alt_baselines, source_artifact, source_run_id
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
            copy_alt_baselines: bool,
        ):
            del copy_alt_baselines, config, draft_id, name, source_artifact, source_run_id
            raise AssertionError("launch should not be called")

        def fork(
            self,
            *,
            run_id: str,
            artifact: str,
            name: str | None,
            config: ManagedRunConfig | None,
            copy_alt_baselines: bool,
        ):
            del config
            assert run_id == parent.id
            assert artifact == "best"
            assert copy_alt_baselines is True
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


async def test_manager_api_reports_only_active_alt_baseline_count(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    run_dir = tmp_path / "runs" / "run-with-alts"
    run = store.create_run(
        run_id="run-with-alts",
        name="Run With Alts",
        config=default_managed_run_config(),
        explicit_run_dir=run_dir,
    )
    store.update_run_status(
        run_id=run.id,
        status="paused",
        started_at="2026-06-13T10:00:00+00:00",
        stopped_at="2026-06-13T10:01:00+00:00",
        message="paused",
    )
    state_path = run_dir / "baselines" / "alt" / "alt-a.state"
    state_path.parent.mkdir(parents=True)
    state_path.write_bytes(b"state")
    store.upsert_run_alt_baseline(
        baseline=TrackSamplingAltBaseline(
            id="alt-a",
            run_id=run.id,
            course_key="mute_city",
            reset_variant_key="gp_race|novice|blue_falcon",
            source_entry_id="mute_city_gp_race_novice_blue_falcon",
            label="active",
            state_path=state_path,
            weight=1.0,
            enabled=True,
            created_at="2026-06-13T10:00:00+00:00",
            updated_at="2026-06-13T10:00:00+00:00",
        )
    )
    store.upsert_run_alt_baseline(
        baseline=TrackSamplingAltBaseline(
            id="alt-missing",
            run_id=run.id,
            course_key="mute_city",
            reset_variant_key="gp_race|novice|blue_falcon",
            source_entry_id="mute_city_gp_race_novice_blue_falcon",
            label="missing",
            state_path=run_dir / "baselines" / "alt" / "missing.state",
            weight=1.0,
            enabled=True,
            created_at="2026-06-13T10:01:00+00:00",
            updated_at="2026-06-13T10:01:00+00:00",
        )
    )

    client = _ApiClient(create_manager_api_app(store, run_launcher=_LauncherStub()))

    response = await client.get("/api/runs")

    assert response.status_code == 200
    payload = response.json()
    assert payload["runs"][0]["active_alt_baseline_count"] == 1


async def test_manager_api_clears_run_alt_baselines_from_database_and_disk(
    tmp_path: Path,
) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    run_dir = tmp_path / "runs" / "run-clear-alts"
    run = store.create_run(
        run_id="run-clear-alts",
        name="Run Clear Alts",
        config=default_managed_run_config(),
        explicit_run_dir=run_dir,
    )
    state_paths = tuple(run_dir / "baselines" / "alt" / f"alt-{index}.state" for index in range(2))
    for index, state_path in enumerate(state_paths):
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_bytes(b"state")
        store.upsert_run_alt_baseline(
            baseline=TrackSamplingAltBaseline(
                id=f"alt-{index}",
                run_id=run.id,
                course_key="mute_city",
                reset_variant_key="gp_race|novice|blue_falcon",
                source_entry_id="mute_city_gp_race_novice_blue_falcon",
                label=f"active {index}",
                state_path=state_path,
                weight=1.0,
                enabled=True,
                created_at=f"2026-06-13T10:0{index}:00+00:00",
                updated_at=f"2026-06-13T10:0{index}:00+00:00",
            )
        )

    client = _client(tmp_path, store=store)
    response = await client.delete(f"/api/runs/{run.id}/track-sampling/alt-baselines")

    assert response.status_code == 200
    payload = response.json()
    assert payload["cleared"] == 2
    assert payload["run"]["active_alt_baseline_count"] == 0
    assert store.get_run_alt_baselines(run.id, include_deleted=True) == ()
    assert all(not state_path.exists() for state_path in state_paths)


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
