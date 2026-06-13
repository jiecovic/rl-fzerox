# tests/core/manager/test_manager_api_drafts_lineages.py
from __future__ import annotations

from pathlib import Path
from typing import Literal

import pytest

import rl_fzerox.apps.run_manager.api.handlers.metrics as manager_api_metrics
from rl_fzerox.apps.run_manager.api import create_manager_api_app
from rl_fzerox.core.manager import (
    ManagedRun,
    ManagedRunConfig,
    ManagerStore,
    default_managed_run_config,
)
from tests.core.manager.manager_api_support import (
    _ApiClient,
    _client,
    _LauncherStub,
)

pytestmark = pytest.mark.anyio


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
            copy_alt_baselines: bool,
        ) -> ManagedRun:
            del name, config, draft_id, source_run_id, source_artifact, copy_alt_baselines
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
