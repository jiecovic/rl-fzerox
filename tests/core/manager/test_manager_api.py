# tests/core/manager/test_manager_api.py
from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from rl_fzerox.apps.run_manager.api import create_manager_api_app
from rl_fzerox.core.manager import ManagerStore, default_managed_run_config


def test_manager_api_lists_default_template(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = client.get("/api/templates")

    assert response.status_code == 200
    payload = response.json()
    assert payload["templates"][0]["id"] == "all_cups_recurrent_ppo"


def test_manager_api_creates_draft(tmp_path: Path) -> None:
    client = _client(tmp_path)
    config = default_managed_run_config().model_dump(mode="json")

    response = client.post("/api/drafts", json={"name": "Draft", "config": config})

    assert response.status_code == 201
    payload = response.json()
    assert payload["draft"]["name"] == "Draft"


def test_manager_api_rejects_invalid_json(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = client.post(
        "/api/drafts",
        content="{",
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 400
    assert "error" in response.json()


def test_manager_api_hides_unstarted_run_records(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    store.create_run(
        name="Created Only",
        config=default_managed_run_config(),
        managed_runs_root=tmp_path / "runs",
    )
    client = TestClient(create_manager_api_app(store))

    response = client.get("/api/runs")

    assert response.status_code == 200
    assert response.json() == {"runs": []}


def _client(tmp_path: Path) -> TestClient:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    return TestClient(create_manager_api_app(store))
