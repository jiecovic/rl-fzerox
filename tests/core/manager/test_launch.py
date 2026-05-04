# tests/core/manager/test_launch.py
from __future__ import annotations

from pathlib import Path

import pytest

from rl_fzerox.apps.run_manager.launch import ManagerRunLauncher
from rl_fzerox.core.manager import ManagerStore, default_managed_run_config
from rl_fzerox.core.training.runs import RUN_LAYOUT


class _RecordingLauncher(ManagerRunLauncher):
    def __init__(self, store: ManagerStore) -> None:
        super().__init__(store)
        self.spawn_calls: list[tuple[str, bool]] = []

    def _spawn_worker(self, *, run_id: str, resume: bool) -> None:
        self.spawn_calls.append((run_id, resume))


def test_resume_relaunches_fork_without_local_checkpoint(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    config = default_managed_run_config()
    source_snapshot_dir = tmp_path / "runs" / "fork-source"
    source_snapshot_dir.mkdir(parents=True)
    store.create_run(
        run_id="parent-run",
        name="Parent Run",
        config=config,
        explicit_run_dir=tmp_path / "runs" / "parent-run",
    )

    run = store.create_run(
        run_id="fork-run",
        name="Fork Run",
        config=config,
        explicit_run_dir=tmp_path / "runs" / "fork-run",
        parent_run_id="parent-run",
        source_run_id="parent-run",
        source_artifact="latest",
        source_snapshot_dir=source_snapshot_dir,
        source_num_timesteps=816_040,
    )
    store.update_run_status(
        run_id=run.id,
        status="failed",
        stopped_at="2026-05-04T14:00:00+00:00",
        message="startup failed",
    )

    launcher = _RecordingLauncher(store)

    resumed = launcher.resume(run_id=run.id)
    events = store.list_recent_run_events((run.id,))

    assert launcher.spawn_calls == [(run.id, False)]
    assert resumed.status == "running"
    assert resumed.started_at is not None
    assert "pinned fork source" in events[run.id][0].message


def test_resume_rebuilds_missing_fork_source_snapshot(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    config = default_managed_run_config()
    parent_dir = tmp_path / "runs" / "parent-run"
    parent_dir.mkdir(parents=True)
    (tmp_path / "core.so").touch()
    (tmp_path / "rom.n64").touch()
    (parent_dir / "train_config.yaml").write_text(
        "\n".join(
            [
                "seed: 7",
                "emulator:",
                f"  core_path: {tmp_path / 'core.so'}",
                f"  rom_path: {tmp_path / 'rom.n64'}",
                "env: {}",
                "reward: {}",
                "policy: {}",
                "curriculum: {}",
                "train:",
                "  algorithm: maskable_ppo",
                "  total_timesteps: 1000",
            ]
        ),
        encoding="utf-8",
    )
    latest_model_path = parent_dir / RUN_LAYOUT.model_artifacts.latest
    latest_policy_path = parent_dir / RUN_LAYOUT.policy_artifacts.latest
    latest_model_path.parent.mkdir(parents=True, exist_ok=True)
    latest_model_path.write_bytes(b"model")
    latest_policy_path.parent.mkdir(parents=True, exist_ok=True)
    latest_policy_path.write_bytes(b"policy")
    latest_policy_path.with_name("policy.metadata.json").write_text(
        '{"num_timesteps": 816040}\n',
        encoding="utf-8",
    )
    parent = store.create_run(
        run_id="parent-run",
        name="Parent Run",
        config=config,
        explicit_run_dir=parent_dir,
    )

    run_dir = tmp_path / "runs" / "fork-run"
    fork_source_dir = run_dir / "fork_source"
    run = store.create_run(
        run_id="fork-run",
        name="Fork Run",
        config=config,
        explicit_run_dir=run_dir,
        lineage_id=parent.lineage_id,
        parent_run_id="parent-run",
        source_run_id="parent-run",
        source_artifact="latest",
        source_snapshot_dir=fork_source_dir,
        source_num_timesteps=816_040,
        lineage_step_offset=816_040,
    )
    store.update_run_status(
        run_id=run.id,
        status="failed",
        stopped_at="2026-05-04T14:00:00+00:00",
        message="startup failed",
    )

    launcher = _RecordingLauncher(store)

    resumed = launcher.resume(run_id=run.id)
    refreshed = store.get_run(run.id)

    assert launcher.spawn_calls == [(run.id, False)]
    assert resumed.status == "running"
    assert resumed.started_at is not None
    assert refreshed is not None
    assert refreshed.source_snapshot_dir is not None
    assert (refreshed.source_snapshot_dir / "train_config.yaml").is_file()
    assert (refreshed.source_snapshot_dir / RUN_LAYOUT.model_artifacts.latest).is_file()
    assert refreshed.source_num_timesteps == 816_040


def test_resume_requires_local_checkpoint_for_root_run(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    config = default_managed_run_config()
    run = store.create_run(
        run_id="root-run",
        name="Root Run",
        config=config,
        explicit_run_dir=tmp_path / "runs" / "root-run",
    )
    store.update_run_status(
        run_id=run.id,
        status="failed",
        stopped_at="2026-05-04T14:00:00+00:00",
        message="startup failed",
    )

    launcher = _RecordingLauncher(store)

    with pytest.raises(ValueError, match="no resumable checkpoint exists"):
        launcher.resume(run_id=run.id)


def test_resume_uses_latest_checkpoint_when_present(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    config = default_managed_run_config()
    run_dir = tmp_path / "runs" / "checkpointed-run"
    run = store.create_run(
        run_id="checkpointed-run",
        name="Checkpointed Run",
        config=config,
        explicit_run_dir=run_dir,
    )
    store.update_run_status(
        run_id=run.id,
        status="stopped",
        stopped_at="2026-05-04T14:00:00+00:00",
        message="stopped cleanly",
    )
    latest_model_path = run_dir / RUN_LAYOUT.model_artifacts.latest
    latest_model_path.parent.mkdir(parents=True, exist_ok=True)
    latest_model_path.write_bytes(b"model")

    launcher = _RecordingLauncher(store)

    resumed = launcher.resume(run_id=run.id)
    events = store.list_recent_run_events((run.id,))

    assert launcher.spawn_calls == [(run.id, True)]
    assert resumed.status == "running"
    assert "latest checkpoint" in events[run.id][0].message
