# tests/core/manager/test_launch.py
from __future__ import annotations

import subprocess
import sys
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
    parent_dir = tmp_path / "runs" / "parent-run"
    source_snapshot_dir = tmp_path / "runs" / "fork-source"
    source_snapshot_dir.mkdir(parents=True)
    latest_model_path = parent_dir / RUN_LAYOUT.model_artifacts.latest
    latest_policy_path = parent_dir / RUN_LAYOUT.policy_artifacts.latest
    train_config_path = parent_dir / RUN_LAYOUT.config_filename
    latest_model_path.parent.mkdir(parents=True, exist_ok=True)
    latest_model_path.write_bytes(b"model")
    latest_policy_path.parent.mkdir(parents=True, exist_ok=True)
    latest_policy_path.write_bytes(b"policy")
    train_config_path.write_text(
        "train:\n  algorithm: maskable_hybrid_recurrent_ppo\n",
        encoding="utf-8",
    )
    latest_policy_path.with_name("policy.metadata.json").write_text(
        '{"num_timesteps": 816040}\n',
        encoding="utf-8",
    )
    store.create_run(
        run_id="parent-run",
        name="Parent Run",
        config=config,
        explicit_run_dir=parent_dir,
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
    train_config_path = parent_dir / RUN_LAYOUT.config_filename
    latest_model_path = parent_dir / RUN_LAYOUT.model_artifacts.latest
    latest_policy_path = parent_dir / RUN_LAYOUT.policy_artifacts.latest
    latest_model_path.parent.mkdir(parents=True, exist_ok=True)
    latest_model_path.write_bytes(b"model")
    latest_policy_path.parent.mkdir(parents=True, exist_ok=True)
    latest_policy_path.write_bytes(b"policy")
    train_config_path.write_text(
        "train:\n  algorithm: maskable_hybrid_recurrent_ppo\n",
        encoding="utf-8",
    )
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
    assert (refreshed.source_snapshot_dir / RUN_LAYOUT.model_artifacts.latest).is_file()
    assert (refreshed.source_snapshot_dir / RUN_LAYOUT.config_filename).read_text(
        encoding="utf-8"
    ) == train_config_path.read_text(encoding="utf-8")
    assert refreshed.source_num_timesteps == 816_040


def test_resume_rebuilds_incomplete_existing_fork_source_snapshot(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    config = default_managed_run_config()
    parent_dir = tmp_path / "runs" / "parent-run"
    parent_dir.mkdir(parents=True)
    latest_model_path = parent_dir / RUN_LAYOUT.model_artifacts.latest
    latest_policy_path = parent_dir / RUN_LAYOUT.policy_artifacts.latest
    train_config_path = parent_dir / RUN_LAYOUT.config_filename
    latest_model_path.parent.mkdir(parents=True, exist_ok=True)
    latest_model_path.write_bytes(b"model")
    latest_policy_path.parent.mkdir(parents=True, exist_ok=True)
    latest_policy_path.write_bytes(b"policy")
    train_config_path.write_text(
        "train:\n  algorithm: maskable_hybrid_recurrent_ppo\n",
        encoding="utf-8",
    )
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
    incomplete_model_path = fork_source_dir / RUN_LAYOUT.model_artifacts.latest
    incomplete_policy_path = fork_source_dir / RUN_LAYOUT.policy_artifacts.latest
    incomplete_model_path.parent.mkdir(parents=True, exist_ok=True)
    incomplete_model_path.write_bytes(b"stale-model")
    incomplete_policy_path.write_bytes(b"stale-policy")
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
    events = store.list_recent_run_events((run.id,))

    assert launcher.spawn_calls == [(run.id, False)]
    assert resumed.status == "running"
    assert refreshed is not None
    assert refreshed.source_snapshot_dir is not None
    assert (refreshed.source_snapshot_dir / RUN_LAYOUT.config_filename).read_text(
        encoding="utf-8"
    ) == train_config_path.read_text(encoding="utf-8")
    assert events[run.id][1].kind == "fork_source_rebuilt"


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


def test_watch_artifact_skips_duplicate_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    config = default_managed_run_config()
    run = store.create_run(
        run_id="watch-run",
        name="Watch Run",
        config=config,
        explicit_run_dir=tmp_path / "runs" / "watch-run",
    )
    launcher = ManagerRunLauncher(store)

    monkeypatch.setattr(
        "rl_fzerox.apps.run_manager.launch.resolve_model_artifact_path",
        lambda *_args, **_kwargs: run.run_dir / RUN_LAYOUT.model_artifacts.latest,
    )
    monkeypatch.setattr(
        "rl_fzerox.apps.run_manager.launch._active_watch_pid",
        lambda **_kwargs: 4321,
    )
    monkeypatch.setattr(
        "rl_fzerox.apps.run_manager.launch.resolve_watch_app_config",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("duplicate watch should not resolve config")
        ),
    )
    monkeypatch.setattr(
        "rl_fzerox.apps.run_manager.launch.subprocess.Popen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("duplicate watch should not spawn a process")
        ),
    )

    launcher.watch_artifact(run_id=run.id, artifact="latest")


def test_watch_artifact_passes_pid_file_to_watch_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    config = default_managed_run_config()
    run = store.create_run(
        run_id="watch-run",
        name="Watch Run",
        config=config,
        explicit_run_dir=tmp_path / "runs" / "watch-run",
    )
    launcher = ManagerRunLauncher(store)
    captured: dict[str, object] = {}

    class _FakeProcess:
        pid = 4321

        def wait(self, timeout: float | None = None) -> int:
            raise subprocess.TimeoutExpired(cmd="watch", timeout=timeout or 0.0)

    def _fake_popen(command: list[str], **_kwargs: object) -> _FakeProcess:
        captured["command"] = command
        return _FakeProcess()

    monkeypatch.setattr(
        "rl_fzerox.apps.run_manager.launch.resolve_model_artifact_path",
        lambda *_args, **_kwargs: run.run_dir / RUN_LAYOUT.model_artifacts.latest,
    )
    monkeypatch.setattr(
        "rl_fzerox.apps.run_manager.launch.resolve_watch_app_config",
        lambda **_kwargs: None,
    )
    pid_path = tmp_path / "manager" / "watch" / f"{run.id}.watch-latest.json"
    monkeypatch.setattr(
        "rl_fzerox.apps.run_manager.launch._manager_watch_pid_path",
        lambda *_args, **_kwargs: pid_path,
    )
    monkeypatch.setattr(
        "rl_fzerox.apps.run_manager.launch.subprocess.Popen",
        _fake_popen,
    )

    status = launcher.watch_artifact(run_id=run.id, artifact="latest")

    assert status == "started"
    assert captured["command"] == [
        sys.executable,
        "-m",
        "rl_fzerox.apps.watch",
        "--manager-db-path",
        str(store.db_path),
        "--managed-run-id",
        run.id,
        "--artifact",
        "latest",
        "--watch-pid-file",
        str(pid_path),
    ]
    assert pid_path.is_file()
