# tests/core/manager/test_manager_api_transfer.py
from __future__ import annotations

import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import rl_fzerox.apps.run_manager.api.handlers.transfer as manager_api_transfer
from rl_fzerox.apps.run_manager.api import create_manager_api_app
from rl_fzerox.core.manager import (
    ManagerStore,
    default_managed_run_config,
)
from rl_fzerox.core.manager.transfer import export_run_bundle
from tests.core.manager.manager_api_support import (
    _LauncherStub,
)

pytestmark = pytest.mark.anyio


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


def test_manager_api_rejects_oversized_run_bundle_upload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    monkeypatch.setattr(manager_api_transfer, "MAX_RUN_BUNDLE_UPLOAD_BYTES", 4)

    with TestClient(create_manager_api_app(store, run_launcher=_LauncherStub())) as client:
        response = client.post(
            "/api/run-imports",
            files={"bundle": ("too-large.zip", b"12345", "application/zip")},
        )

    assert response.status_code == 413
    assert response.json()["error"] == "run bundle upload exceeds 4 bytes"
