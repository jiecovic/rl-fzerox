# tests/core/manager/test_manager_store.py
from __future__ import annotations

from pathlib import Path

from rl_fzerox.core.manager import ManagerStore, default_managed_run_config


def test_manager_store_seeds_default_template(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "runs.db")

    templates = store.list_templates()

    assert len(templates) == 1
    assert templates[0].id == "all_cups_recurrent_ppo"
    assert templates[0].config == default_managed_run_config()


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


def test_manager_store_deletes_draft(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    draft = store.create_draft(
        name="Delete Me",
        config=default_managed_run_config(),
    )

    assert store.delete_draft(draft.id)

    assert store.list_drafts() == ()


def test_manager_store_creates_run_record_without_filesystem_artifacts(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    config = default_managed_run_config()

    run = store.create_run(
        name="Started Later",
        config=config,
        managed_runs_root=tmp_path / "managed_runs",
    )

    assert store.list_runs()[0].id == run.id
    assert not run.run_dir.exists()
