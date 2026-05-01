# tests/core/manager/test_manager_store.py
from __future__ import annotations

import json
from pathlib import Path

from rl_fzerox.core.manager import ManagerStore, default_managed_run_config
from rl_fzerox.core.manager.manifest import MANIFEST_WARNING


def test_manager_store_seeds_default_template(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "runs.db")

    templates = store.list_templates()

    assert len(templates) == 1
    assert templates[0].id == "all_cups_recurrent_ppo"
    assert templates[0].config == default_managed_run_config()


def test_manager_store_creates_immutable_run_with_manifest(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    config = default_managed_run_config().model_copy(update={"seed": 321})

    run = store.create_run(
        name="Prototype Run",
        config=config,
        managed_runs_root=tmp_path / "managed_runs",
    )

    runs = store.list_runs()
    assert len(runs) == 1
    assert runs[0].id == run.id
    assert runs[0].name == "Prototype Run"
    assert runs[0].status == "created"
    assert runs[0].config == config

    manifest = json.loads((run.run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["run_id"] == run.id
    assert manifest["config_hash"] == run.config_hash
    assert manifest["warning"] == MANIFEST_WARNING
    assert (run.run_dir / "manager_config.yaml").is_file()
