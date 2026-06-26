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

    response = await client.post("/api/checkpoints/catalog/blue-falcon-fine-tuned/v1/install")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "installed"
    assert payload["checkpoint"]["id"] == "blue-falcon-fine-tuned-v1"
    assert payload["checkpoint"]["run_id"] == "checkpoint-blue-falcon-fine-tuned-v1"
    assert payload["checkpoint"]["run"]["id"] == "checkpoint-blue-falcon-fine-tuned-v1"
    assert payload["checkpoint"]["run"]["status"] == "archived"
    assert payload["checkpoint"]["config"]["version"] == 1
    assert payload["checkpoint"]["import_dir"].endswith(
        "manager/checkpoints/blue-falcon-fine-tuned/v1"
    )
    assert store.get_published_checkpoint("blue-falcon-fine-tuned-v1") is not None


async def test_manager_api_catalog_includes_installed_checkpoint_run(
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
    install_response = await client.post(
        "/api/checkpoints/catalog/blue-falcon-fine-tuned/v1/install"
    )
    assert install_response.status_code == 200

    response = await client.get("/api/checkpoints/catalog")

    assert response.status_code == 200
    checkpoint = response.json()["installed_checkpoints"][0]
    assert checkpoint["run_id"] == "checkpoint-blue-falcon-fine-tuned-v1"
    assert checkpoint["run"]["id"] == checkpoint["run_id"]
    assert checkpoint["run"]["status"] == "archived"


async def test_manager_api_does_not_expose_checkpoint_fork_route(tmp_path: Path) -> None:
    client = _client(tmp_path, store=ManagerStore(tmp_path / "manager" / "runs.db"))

    response = await client.post(
        "/api/checkpoints/blue-falcon-fine-tuned-v1/fork",
        json={"name": "Blue Falcon Fork"},
    )

    assert response.status_code == 404


async def test_manager_api_deletes_installed_checkpoint(
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
    install_response = await client.post(
        "/api/checkpoints/catalog/blue-falcon-fine-tuned/v1/install"
    )
    assert install_response.status_code == 200
    checkpoint = store.get_published_checkpoint("blue-falcon-fine-tuned-v1")
    assert checkpoint is not None
    assert checkpoint.import_dir.is_dir()

    response = await client.delete("/api/checkpoints/blue-falcon-fine-tuned-v1")

    assert response.status_code == 200
    assert response.json() == {"deleted": True}
    assert store.get_published_checkpoint("blue-falcon-fine-tuned-v1") is None
    assert not checkpoint.import_dir.exists()


async def test_manager_api_rejects_checkpoint_delete_when_save_game_uses_it(
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
    install_response = await client.post(
        "/api/checkpoints/catalog/blue-falcon-fine-tuned/v1/install"
    )
    assert install_response.status_code == 200
    checkpoint = store.get_published_checkpoint("blue-falcon-fine-tuned-v1")
    assert checkpoint is not None
    save_game = store.create_save_game(name="Career", save_games_root=tmp_path / "saves")
    store.upsert_save_course_setup(
        save_game_id=save_game.id,
        policy_source_kind="checkpoint",
        policy_source_id=checkpoint.id,
        policy_artifact=checkpoint.source_artifact,
        cup_id="jack",
        course_id="mute_city",
    )

    response = await client.delete("/api/checkpoints/blue-falcon-fine-tuned-v1")

    assert response.status_code == 400
    assert (
        response.json()["error"] == "remove save-game course setups that still use this checkpoint"
    )
    assert store.get_published_checkpoint("blue-falcon-fine-tuned-v1") is not None
    assert checkpoint.import_dir.is_dir()


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

    first_response = await client.post("/api/checkpoints/catalog/blue-falcon-fine-tuned/v1/install")
    assert first_response.status_code == 200

    def fail_download(*_args: object, **_kwargs: object) -> Path:
        raise AssertionError("download should not be called for installed checkpoint")

    monkeypatch.setattr(manager_api_checkpoints, "_download_catalog_bundle", fail_download)

    second_response = await client.post(
        "/api/checkpoints/catalog/blue-falcon-fine-tuned/v1/install"
    )

    assert second_response.status_code == 200
    payload = second_response.json()
    assert payload["status"] == "already_installed"
    assert payload["checkpoint"]["run_id"] == "checkpoint-blue-falcon-fine-tuned-v1"
    assert payload["checkpoint"]["run"]["status"] == "archived"


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
            "checkpoint": manifest.checkpoint.model_copy(update={"id": "blue-falcon-fine-tuned"})
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
