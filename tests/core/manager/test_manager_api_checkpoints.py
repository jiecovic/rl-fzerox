# tests/core/manager/test_manager_api_checkpoints.py
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

import rl_fzerox.apps.run_manager.api.handlers.checkpoints as manager_api_checkpoints
from rl_fzerox.core.manager import ManagerStore
from rl_fzerox.core.manager.checkpoints import (
    CheckpointCatalog,
    CheckpointCatalogBundle,
    CheckpointCatalogEntry,
    serialize_checkpoint_catalog_json,
)
from tests.core.manager.manager_api_support import _client
from tests.core.manager.test_manager_store_checkpoints import _manifest, _payloads, _write_bundle

pytestmark = pytest.mark.anyio


async def test_manager_api_lists_checkpoint_catalog(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle_path, catalog_path = _write_catalog_bundle(tmp_path)
    del bundle_path
    monkeypatch.setattr(
        manager_api_checkpoints,
        "default_checkpoint_catalog_path",
        lambda: catalog_path,
    )
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    client = _client(tmp_path, store=store)

    response = await client.get("/api/checkpoints/catalog")

    assert response.status_code == 200
    payload = response.json()
    assert payload["catalog"]["schema_version"] == 1
    assert payload["entries"][0]["id"] == "blue-falcon-fine-tuned"
    assert payload["entries"][0]["installed_checkpoint_id"] is None
    assert payload["installed_checkpoints"] == []


async def test_manager_api_installs_checkpoint_catalog_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle_path, catalog_path = _write_catalog_bundle(tmp_path)
    monkeypatch.setattr(
        manager_api_checkpoints,
        "default_checkpoint_catalog_path",
        lambda: catalog_path,
    )
    monkeypatch.setattr(
        manager_api_checkpoints,
        "_download_catalog_bundle",
        lambda entry, *, download_dir: bundle_path,
    )
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    client = _client(tmp_path, store=store)

    response = await client.post(
        "/api/checkpoints/catalog/blue-falcon-fine-tuned/v1/install"
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "installed"
    assert payload["checkpoint"]["id"] == "blue-falcon-fine-tuned-v1"
    assert store.get_published_checkpoint("blue-falcon-fine-tuned-v1") is not None


async def test_manager_api_checkpoint_catalog_install_is_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle_path, catalog_path = _write_catalog_bundle(tmp_path)
    monkeypatch.setattr(
        manager_api_checkpoints,
        "default_checkpoint_catalog_path",
        lambda: catalog_path,
    )
    monkeypatch.setattr(
        manager_api_checkpoints,
        "_download_catalog_bundle",
        lambda entry, *, download_dir: bundle_path,
    )
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    client = _client(tmp_path, store=store)

    first_response = await client.post(
        "/api/checkpoints/catalog/blue-falcon-fine-tuned/v1/install"
    )
    assert first_response.status_code == 200

    def fail_download(*_args: object, **_kwargs: object) -> Path:
        raise AssertionError("download should not be called for installed checkpoint")

    monkeypatch.setattr(manager_api_checkpoints, "_download_catalog_bundle", fail_download)

    second_response = await client.post(
        "/api/checkpoints/catalog/blue-falcon-fine-tuned/v1/install"
    )

    assert second_response.status_code == 200
    assert second_response.json()["status"] == "already_installed"


async def test_manager_api_checkpoint_catalog_returns_404_for_unknown_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _bundle_path, catalog_path = _write_catalog_bundle(tmp_path)
    monkeypatch.setattr(
        manager_api_checkpoints,
        "default_checkpoint_catalog_path",
        lambda: catalog_path,
    )
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    client = _client(tmp_path, store=store)

    response = await client.post("/api/checkpoints/catalog/missing/v1/install")

    assert response.status_code == 404
    assert response.json()["error"] == "checkpoint catalog entry not found"


def _write_catalog_bundle(tmp_path: Path) -> tuple[Path, Path]:
    payloads = _payloads()
    manifest = _manifest(payloads)
    manifest = manifest.model_copy(
        update={
            "checkpoint": manifest.checkpoint.model_copy(
                update={"id": "blue-falcon-fine-tuned"}
            )
        }
    )
    bundle_path = tmp_path / "blue-falcon.zip"
    _write_bundle(bundle_path, manifest=manifest, payloads=payloads)
    catalog = CheckpointCatalog(
        updated_at="2026-06-26T12:00:00+00:00",
        entries=(
            CheckpointCatalogEntry(
                id=manifest.checkpoint.id,
                version=manifest.checkpoint.version,
                bundle=CheckpointCatalogBundle(
                    url="https://example.invalid/blue-falcon.zip",
                    filename="blue-falcon.zip",
                    size_bytes=bundle_path.stat().st_size,
                    sha256=hashlib.sha256(bundle_path.read_bytes()).hexdigest(),
                ),
                manifest=manifest,
            ),
        ),
    )
    catalog_path = tmp_path / "published_checkpoints.json"
    catalog_path.write_text(serialize_checkpoint_catalog_json(catalog), encoding="utf-8")
    return bundle_path, catalog_path
