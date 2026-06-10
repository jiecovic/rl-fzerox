# tests/core/manager/test_manager_store_drafts_lineages.py
from __future__ import annotations

from pathlib import Path
from typing import Literal

import pytest

import rl_fzerox.core.manager.artifacts.filesystem as filesystem_ops_module
import rl_fzerox.core.manager.registry.drafts.fork_sources as draft_fork_sources
from rl_fzerox.core.manager import (
    ManagerStore,
    default_managed_run_config,
)
from rl_fzerox.core.manager.artifacts.filesystem import FilesystemOperation
from tests.core.manager.manager_store_support import (
    _filesystem_operation_count,
)

SnapshotKind = Literal["run", "draft", "template", "import"]


def test_manager_store_saves_draft_without_filesystem_artifacts(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    config = default_managed_run_config().model_copy(update={"seed": 321})

    draft = store.create_draft(
        name="Prototype Run",
        config=config,
    )

    drafts = store.list_drafts()
    assert len(drafts) == 1
    assert drafts[0].id == draft.id
    assert drafts[0].name == "Prototype Run"
    assert drafts[0].config == config
    assert not (tmp_path / "managed_runs").exists()


def test_manager_store_pins_and_cleans_fork_draft_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    config = default_managed_run_config()
    source_run = store.create_run(
        run_id="parent-run",
        name="Parent Run",
        config=config,
        explicit_run_dir=tmp_path / "runs" / "parent-run",
    )

    def fake_snapshot_fork_source(
        *,
        source_run_dir: Path,
        artifact: str,
        destination_dir: Path,
    ) -> int:
        assert source_run_dir == source_run.run_dir
        assert artifact == "latest"
        destination_dir.mkdir(parents=True, exist_ok=True)
        return 123_456

    monkeypatch.setattr(
        draft_fork_sources,
        "snapshot_fork_source",
        fake_snapshot_fork_source,
    )

    draft = store.create_draft(
        name="Pinned Fork",
        config=config,
        source_run_id=source_run.id,
        source_artifact="latest",
    )

    assert draft.source_snapshot_dir is not None
    assert draft.source_snapshot_dir.is_dir()
    assert draft.source_num_timesteps == 123_456

    assert store.delete_draft(draft.id)
    assert not draft.source_snapshot_dir.exists()


def test_manager_store_persists_run_lineage_fields(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    config = default_managed_run_config()
    store.create_run(
        run_id="parent-run",
        name="Parent Run",
        config=config,
        explicit_run_dir=tmp_path / "runs" / "parent-run",
    )

    run = store.create_run(
        run_id="child-run",
        name="Child Run",
        config=config,
        explicit_run_dir=tmp_path / "runs" / "child-run",
        lineage_step_offset=816_040,
        parent_run_id="parent-run",
        source_run_id="parent-run",
        source_artifact="best",
        source_num_timesteps=816_040,
    )

    loaded = store.get_run(run.id)

    assert loaded is not None
    assert loaded.lineage_id == "parent-run"
    assert loaded.lineage_step_offset == 816_040
    assert loaded.parent_run_id == "parent-run"
    assert loaded.source_run_id == "parent-run"
    assert loaded.source_artifact == "best"
    assert loaded.source_num_timesteps == 816_040


def test_manager_store_assigns_lineage_groups_and_builds_tensorboard_views(
    tmp_path: Path,
) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    config = default_managed_run_config()
    run = store.create_run(
        run_id="run-a",
        name="Medium CNN",
        config=config,
        explicit_run_dir=tmp_path / "runs" / "run-a" / "run-a",
    )
    tensorboard_dir = run.run_dir / "tensorboard"
    tensorboard_dir.mkdir(parents=True)
    store.update_run_status(run_id=run.id, status="stopped", message="stopped")

    group_names = store.update_lineage_groups(
        lineage_id=run.lineage_id,
        group_names=("76x108 CNN sweep", "Current ablations"),
    )
    views = store.rebuild_tensorboard_views()
    loaded = store.get_run(run.id)

    assert group_names == ("76x108 CNN sweep", "Current ablations")
    assert loaded is not None
    assert loaded.lineage_groups == ("76x108 CNN sweep", "Current ablations")
    assert views[0].slug == "76x108-cnn-sweep"
    links = tuple(
        path
        for path in (store.tensorboard_views_root() / "76x108-cnn-sweep").rglob("*__run-a")
        if path.is_symlink()
    )
    second_group_links = tuple(
        path
        for path in (store.tensorboard_views_root() / "current-ablations").rglob("*__run-a")
        if path.is_symlink()
    )
    assert len(links) == 1
    assert len(second_group_links) == 1
    assert links[0].is_symlink()
    assert links[0].resolve() == tensorboard_dir.resolve()


def test_manager_store_deletes_draft(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    draft = store.create_draft(
        name="Delete Me",
        config=default_managed_run_config(),
    )

    assert store.delete_draft(draft.id)

    assert store.list_drafts() == ()


def test_manager_store_allows_draft_name_used_by_run(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    config = default_managed_run_config()
    store.create_run(name="Shared Name", config=config, managed_runs_root=tmp_path / "managed_runs")
    draft = store.create_draft(name="Shared Name", config=config)

    assert draft.name == "Shared Name"


def test_manager_store_allows_run_name_used_by_draft(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    config = default_managed_run_config()
    store.create_draft(name="Shared Name", config=config)
    run = store.create_run(
        name="Shared Name",
        config=config,
        managed_runs_root=tmp_path / "managed_runs",
    )

    assert run.name == "Shared Name"


def test_manager_store_renames_run_without_mutating_config(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    run = store.create_run(
        name="Old Name",
        config=default_managed_run_config(),
        managed_runs_root=tmp_path / "runs",
    )

    renamed = store.update_run_name(run_id=run.id, name="New Name")

    assert renamed is not None
    assert renamed.name == "New Name"
    assert renamed.config == run.config


def test_manager_store_allows_duplicate_run_names(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    config = default_managed_run_config()
    first = store.create_run(name="Shared Run", config=config, managed_runs_root=tmp_path / "runs")
    second = store.create_run(
        name="Shared Run",
        config=config,
        managed_runs_root=tmp_path / "runs",
    )

    assert first.name == "Shared Run"
    assert second.name == "Shared Run"
    assert first.id != second.id


def test_manager_store_allows_renaming_run_to_existing_run_name(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    config = default_managed_run_config()
    store.create_run(name="Existing Run", config=config, managed_runs_root=tmp_path / "runs")
    target = store.create_run(name="Target Run", config=config, managed_runs_root=tmp_path / "runs")

    renamed = store.update_run_name(run_id=target.id, name="Existing Run")

    assert renamed is not None
    assert renamed.name == "Existing Run"


def test_manager_store_deletes_full_lineage(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    config = default_managed_run_config()
    parent = store.create_run(
        run_id="parent-run",
        name="Parent Run",
        config=config,
        managed_runs_root=tmp_path / "runs",
    )
    child = store.create_run(
        run_id="child-run",
        name="Child Run",
        config=config,
        managed_runs_root=tmp_path / "runs",
        lineage_id=parent.lineage_id,
        parent_run_id=parent.id,
        source_run_id=parent.id,
        source_artifact="latest",
        source_num_timesteps=456,
    )
    parent.run_dir.mkdir(parents=True)
    child.run_dir.mkdir(parents=True)

    assert store.delete_lineage(parent.lineage_id) is True
    assert store.get_run(parent.id) is None
    assert store.get_run(child.id) is None
    assert not parent.run_dir.exists()


def test_manager_store_delete_lineage_defers_failed_filesystem_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    config = default_managed_run_config()
    parent = store.create_run(
        run_id="parent-run",
        name="Parent Run",
        config=config,
        managed_runs_root=tmp_path / "runs",
    )
    child = store.create_run(
        run_id="child-run",
        name="Child Run",
        config=config,
        managed_runs_root=tmp_path / "runs",
        lineage_id=parent.lineage_id,
        parent_run_id=parent.id,
        source_run_id=parent.id,
        source_artifact="latest",
        source_num_timesteps=456,
    )
    parent.run_dir.mkdir(parents=True)
    child.run_dir.mkdir(parents=True)
    original_apply = filesystem_ops_module.apply_filesystem_operation

    def fail_delete(operation: FilesystemOperation) -> bool:
        raise OSError("filesystem busy")

    monkeypatch.setattr(filesystem_ops_module, "apply_filesystem_operation", fail_delete)

    assert store.delete_lineage(parent.lineage_id) is True
    assert store.get_run(parent.id) is None
    assert store.get_run(child.id) is None
    assert parent.run_dir.exists()
    assert _filesystem_operation_count(store) >= 2

    monkeypatch.setattr(filesystem_ops_module, "apply_filesystem_operation", original_apply)
    recovered = ManagerStore(store.db_path)
    recovered.initialize()

    assert not parent.run_dir.exists()
    assert _filesystem_operation_count(store) == 0
