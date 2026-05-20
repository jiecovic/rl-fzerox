# tests/core/manager/test_transfer.py
from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from rl_fzerox.core.manager import ManagerStore, default_managed_run_config
from rl_fzerox.core.manager.transfer import RunBundleError, export_run_bundle, import_run_bundle


def test_run_bundle_export_import_rewrites_paths(tmp_path: Path) -> None:
    source_store = ManagerStore(tmp_path / "source" / "manager" / "runs.db")
    source_run_dir = tmp_path / "source" / "runs" / "lineage-a" / "run-a"
    config = default_managed_run_config().model_copy(update={"seed": 123})
    run = source_store.create_run(
        run_id="run-a",
        name="Portable Run",
        config=config,
        explicit_run_dir=source_run_dir,
        lineage_id="lineage-a",
    )
    source_run_dir.mkdir(parents=True)
    (source_run_dir / "checkpoints" / "latest").mkdir(parents=True)
    (source_run_dir / "checkpoints" / "latest" / "model.zip").write_bytes(b"model")
    (source_run_dir / "train_config.yaml").write_text(
        "\n".join(
            (
                f"runtime_dir: {source_run_dir / 'runtime'}",
                f"output_root: {source_run_dir.parent}",
                "run_name: run-a",
                f"explicit_run_dir: {source_run_dir}",
            )
        ),
        encoding="utf-8",
    )
    source_store.update_run_status(run_id=run.id, status="stopped", message="stopped")
    source_store.upsert_run_runtime(
        run_id=run.id,
        total_timesteps=10_000,
        num_timesteps=2_000,
        progress_fraction=0.2,
        updated_at="2026-05-20T10:00:00+00:00",
        fps=6.5,
    )
    source_store.update_lineage_groups(lineage_id=run.lineage_id, group_names=("portable",))

    bundle_path = export_run_bundle(
        store=source_store,
        run_id=run.id,
        output_path=tmp_path / "run-a.zip",
    )

    target_store = ManagerStore(tmp_path / "target" / "manager" / "runs.db")
    target_runs_root = tmp_path / "target" / "runs"
    result = import_run_bundle(
        store=target_store,
        bundle_path=bundle_path,
        run_id="imported-run",
        managed_runs_root=target_runs_root,
    )

    imported = target_store.get_run("imported-run")
    assert imported is not None
    assert result.run_dir == str(target_runs_root / "lineage-a" / "imported-run")
    assert imported.run_dir == target_runs_root / "lineage-a" / "imported-run"
    assert imported.name == "Portable Run"
    assert imported.status == "stopped"
    assert imported.runtime is not None
    assert imported.runtime.num_timesteps == 2_000
    assert imported.lineage_groups == ("portable",)
    assert (imported.run_dir / "checkpoints" / "latest" / "model.zip").read_bytes() == b"model"
    assert str(source_run_dir) not in (imported.run_dir / "train_config.yaml").read_text(
        encoding="utf-8"
    )
    assert str(imported.run_dir) in (imported.run_dir / "train_config.yaml").read_text(
        encoding="utf-8"
    )
    assert "run_name: imported-run" in (imported.run_dir / "train_config.yaml").read_text(
        encoding="utf-8"
    )


def test_run_bundle_import_rejects_path_traversal(tmp_path: Path) -> None:
    bundle_path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(bundle_path, mode="w") as archive:
        archive.writestr(
            "run_export.json",
            """
            {
              "format_name": "rl-fzerox-run-bundle",
              "schema_version": 1,
              "exported_at": "2026-05-20T10:00:00+00:00",
              "project_root": "/old/project",
              "run": {
                "id": "unsafe-run",
                "name": "Unsafe",
                "status": "stopped",
                "config": {},
                "run_dir": "/old/project/local/runs/unsafe-run/unsafe-run",
                "lineage_id": "unsafe-run",
                "created_at": "2026-05-20T10:00:00+00:00"
              },
              "files": []
            }
            """,
        )
        archive.writestr("../evil.txt", "bad")

    with pytest.raises(RunBundleError, match="unsafe archive member"):
        import_run_bundle(
            store=ManagerStore(tmp_path / "manager" / "runs.db"),
            bundle_path=bundle_path,
            managed_runs_root=tmp_path / "runs",
        )


def test_run_bundle_import_does_not_delete_existing_target_dir(tmp_path: Path) -> None:
    source_store = ManagerStore(tmp_path / "source" / "manager" / "runs.db")
    source_run_dir = tmp_path / "source" / "runs" / "run-a" / "run-a"
    run = source_store.create_run(
        run_id="run-a",
        name="Portable Run",
        config=default_managed_run_config(),
        explicit_run_dir=source_run_dir,
    )
    source_run_dir.mkdir(parents=True)
    (source_run_dir / "train_config.yaml").write_text("run_name: run-a\n", encoding="utf-8")
    source_store.update_run_status(run_id=run.id, status="stopped", message="stopped")
    bundle_path = export_run_bundle(
        store=source_store,
        run_id=run.id,
        output_path=tmp_path / "run-a.zip",
    )
    existing_dir = tmp_path / "target" / "runs" / "run-a" / "run-a"
    existing_dir.mkdir(parents=True)
    sentinel = existing_dir / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(RunBundleError, match="target run directory already exists"):
        import_run_bundle(
            store=ManagerStore(tmp_path / "target" / "manager" / "runs.db"),
            bundle_path=bundle_path,
            managed_runs_root=tmp_path / "target" / "runs",
        )

    assert sentinel.read_text(encoding="utf-8") == "keep"
