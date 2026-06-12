# tests/core/manager/test_launch.py
from __future__ import annotations

import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import BinaryIO

import pytest

from rl_fzerox.apps.run_manager.launch import ManagerRunLauncher
from rl_fzerox.apps.run_manager.launching.save_games import active_career_mode_runner_pid
from rl_fzerox.apps.run_manager.launching.watch import active_watch_pid, watch_failure_detail
from rl_fzerox.core.domain.courses import BUILT_IN_COURSES
from rl_fzerox.core.manager import ManagedRun, ManagerStore, default_managed_run_config
from rl_fzerox.core.training.runs import RUN_LAYOUT


class _RecordingLauncher(ManagerRunLauncher):
    def __init__(self, store: ManagerStore) -> None:
        super().__init__(store)
        self.spawn_calls: list[tuple[str, bool]] = []

    def _spawn_worker(self, *, run_id: str, resume: bool) -> None:
        self.spawn_calls.append((run_id, resume))


def test_launch_allows_unsaved_fork_source(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    config = default_managed_run_config()
    source_run = store.create_run(
        run_id="parent-run",
        name="Parent Run",
        config=config,
        explicit_run_dir=tmp_path / "runs" / "parent-run",
    )

    class _ForkRecordingLauncher(ManagerRunLauncher):
        def __init__(self, store: ManagerStore) -> None:
            super().__init__(store)
            self.fork_calls: list[tuple[str, str, str | None, object | None]] = []

        def fork(
            self,
            *,
            run_id: str,
            artifact: str,
            name: str | None = None,
            config: object | None = None,
            exclude_draft_id: str | None = None,
            source_snapshot_dir: Path | None = None,
            source_num_timesteps: int | None = None,
        ) -> ManagedRun:
            del exclude_draft_id, source_num_timesteps, source_snapshot_dir
            self.fork_calls.append((run_id, artifact, name, config))
            return source_run

    launcher = _ForkRecordingLauncher(store)

    launched = launcher.launch(
        name="Forked Child",
        config=config,
        draft_id=None,
        source_run_id=source_run.id,
        source_artifact="best",
    )

    assert launched == source_run
    assert launcher.fork_calls == [(source_run.id, "best", "Forked Child", config)]


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
        "rl_fzerox.apps.run_manager.launching.watch.resolve_model_artifact_path",
        lambda *_args, **_kwargs: run.run_dir / RUN_LAYOUT.model_artifacts.latest,
    )
    monkeypatch.setattr(
        "rl_fzerox.apps.run_manager.launching.watch.active_watch_pid",
        lambda **_kwargs: 4321,
    )
    monkeypatch.setattr(
        "rl_fzerox.apps.run_manager.launching.watch.resolve_watch_app_config",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("duplicate watch should not resolve config")
        ),
    )
    monkeypatch.setattr(
        "rl_fzerox.apps.run_manager.launching.watch.subprocess.Popen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("duplicate watch should not spawn a process")
        ),
    )

    launcher.watch_artifact(
        run_id=run.id,
        artifact="latest",
        device="cuda",
        renderer=None,
        deterministic_policy=True,
    )


def test_active_watch_pid_clears_stale_viewer_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    run = store.create_run(
        run_id="watch-run",
        name="Watch Run",
        config=default_managed_run_config(),
        explicit_run_dir=tmp_path / "runs" / "watch-run",
    )
    lease_id = store.viewer_lease_id(
        kind="run_watch",
        owner_id=run.id,
        qualifier="latest",
    )
    store.upsert_viewer_lease(
        lease_id=lease_id,
        kind="run_watch",
        owner_id=run.id,
        pid=os.getpid(),
        qualifier="latest",
    )
    assert store.heartbeat_viewer_lease(
        lease_id=lease_id,
        pid=os.getpid(),
        heartbeat_at=_stale_heartbeat_at(),
    )
    monkeypatch.setattr(
        "rl_fzerox.apps.run_manager.launching.watch.watch_process_matches",
        lambda **_kwargs: True,
    )

    assert (
        active_watch_pid(
            store=store,
            lease_id=lease_id,
            run_id=run.id,
            run_dir=run.run_dir,
            artifact="latest",
        )
        is None
    )
    assert store.get_viewer_lease(lease_id) is None


def test_watch_artifact_passes_viewer_lease_to_watch_process(
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
    log_path = tmp_path / "logs" / "watch-latest.log"
    log_path.parent.mkdir(parents=True)
    log_path.write_text("stale traceback\n", encoding="utf-8")
    captured: dict[str, object] = {}

    class _FakeProcess:
        pid = 4321

        def wait(self, timeout: float | None = None) -> int:
            if timeout is None:
                return 0
            raise subprocess.TimeoutExpired(cmd="watch", timeout=timeout or 0.0)

    def _fake_popen(
        command: list[str],
        *,
        cwd: Path,
        stdin: object,
        stdout: BinaryIO,
        stderr: object,
        start_new_session: bool,
    ) -> _FakeProcess:
        del cwd, stdin, stderr, start_new_session
        captured["command"] = command
        stdout.write(b"watch started\n")
        return _FakeProcess()

    monkeypatch.setattr(
        "rl_fzerox.apps.run_manager.launching.watch.resolve_model_artifact_path",
        lambda *_args, **_kwargs: run.run_dir / RUN_LAYOUT.model_artifacts.latest,
    )
    monkeypatch.setattr(
        "rl_fzerox.apps.run_manager.launching.watch.resolve_watch_app_config",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        "rl_fzerox.apps.run_manager.launching.watch.manager_watch_log_path",
        lambda run_id, *, artifact: log_path,
    )
    monkeypatch.setattr(
        "rl_fzerox.apps.run_manager.launching.watch.subprocess.Popen",
        _fake_popen,
    )

    status = launcher.watch_artifact(
        run_id=run.id,
        artifact="latest",
        device="cuda",
        renderer="angrylion",
        deterministic_policy=False,
    )

    assert status == "started"
    lease_id = store.viewer_lease_id(
        kind="run_watch",
        owner_id=run.id,
        qualifier="latest",
    )
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
        "--viewer-lease-id",
        lease_id,
        "--",
        "watch.device=cuda",
        "watch.deterministic_policy=false",
        "emulator.renderer=angrylion",
    ]
    lease = store.get_viewer_lease(lease_id)
    assert lease is not None
    assert lease.kind == "run_watch"
    assert lease.owner_id == run.id
    assert lease.qualifier == "latest"
    assert lease.pid == 4321
    log_text = log_path.read_text(encoding="utf-8")
    assert "stale traceback" not in log_text
    assert "# launched_at=" in log_text
    assert "# command=" in log_text
    assert "watch started" in log_text


def test_watch_failure_detail_ignores_launch_log_headers(tmp_path: Path) -> None:
    log_path = tmp_path / "watch.log"
    log_path.write_text(
        "# launched_at=2026-06-08T12:00:00+00:00\n# command=python -m rl_fzerox.apps.watch\n\n",
        encoding="utf-8",
    )

    assert watch_failure_detail(log_path) is None

    log_path.write_text(
        "# launched_at=2026-06-08T12:00:00+00:00\nRuntimeError: failed for real\n",
        encoding="utf-8",
    )

    assert watch_failure_detail(log_path) == "RuntimeError: failed for real"


def test_start_career_mode_passes_viewer_lease_and_runtime_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    save_game = store.create_save_game(
        name="Unlock Save",
        save_games_root=tmp_path / "career-saves",
    )
    run = store.create_run(
        name="Unlock Policy",
        config=default_managed_run_config(),
        managed_runs_root=tmp_path / "runs",
    )
    _configure_gp_cup(store, save_game_id=save_game.id, run_id=run.id, cup_id="jack")
    launcher = ManagerRunLauncher(store)
    log_path = tmp_path / "logs" / f"{save_game.id}.log"
    recording_path = tmp_path / "recordings" / "career.mkv"
    log_path.parent.mkdir(parents=True)
    log_path.write_text("stale career failure\n", encoding="utf-8")
    captured: dict[str, object] = {}

    class _FakeProcess:
        pid = 8765

        def wait(self, timeout: float | None = None) -> int:
            if timeout is None:
                return 0
            raise subprocess.TimeoutExpired(cmd="career-mode", timeout=timeout or 0.0)

    def _fake_popen(
        command: list[str],
        *,
        cwd: Path,
        stdin: object,
        stdout: BinaryIO,
        stderr: object,
        start_new_session: bool,
    ) -> _FakeProcess:
        del cwd, stdin, stderr, start_new_session
        captured["command"] = command
        stdout.write(b"career mode started\n")
        return _FakeProcess()

    monkeypatch.setattr(
        "rl_fzerox.apps.run_manager.launching.save_games.manager_career_mode_log_path",
        lambda save_game_id: log_path,
    )
    monkeypatch.setattr(
        "rl_fzerox.apps.run_manager.launching.save_games.subprocess.Popen",
        _fake_popen,
    )

    status = launcher.start_career_mode(
        save_game_id=save_game.id,
        device="cpu",
        renderer="angrylion",
        attempt_seed=1234,
        deterministic_policy=False,
        recording_enabled=True,
        recording_path=recording_path,
    )

    assert status == "started"
    lease_id = store.viewer_lease_id(kind="career_mode", owner_id=save_game.id)
    attempts = store.list_save_attempts(save_game.id)
    assert len(attempts) == 1
    assert attempts[0].cup_id == "jack"
    assert captured["command"] == [
        sys.executable,
        "-m",
        "rl_fzerox.apps.career_mode",
        "--manager-db-path",
        str(store.db_path),
        "--save-game-id",
        save_game.id,
        "--save-attempt-id",
        attempts[0].id,
        "--viewer-lease-id",
        lease_id,
        "--attempt-seed",
        "1234",
        "--policy-mode",
        "stochastic",
        "--",
        "watch.device=cpu",
        "watch.deterministic_policy=false",
        "emulator.renderer=angrylion",
        "watch.recording.enabled=true",
        f"watch.recording.path={recording_path}",
    ]
    lease = store.get_viewer_lease(lease_id)
    assert lease is not None
    assert lease.kind == "career_mode"
    assert lease.owner_id == save_game.id
    assert lease.qualifier is None
    assert lease.pid == 8765
    log_text = log_path.read_text(encoding="utf-8")
    assert "stale career failure" not in log_text
    assert "# launched_at=" in log_text
    assert "# command=" in log_text
    assert "career mode started" in log_text


def test_active_career_mode_runner_pid_clears_stale_viewer_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    save_game = store.create_save_game(
        name="Unlock Save",
        save_games_root=tmp_path / "career-saves",
    )
    lease_id = store.viewer_lease_id(kind="career_mode", owner_id=save_game.id)
    store.upsert_viewer_lease(
        lease_id=lease_id,
        kind="career_mode",
        owner_id=save_game.id,
        pid=os.getpid(),
    )
    assert store.heartbeat_viewer_lease(
        lease_id=lease_id,
        pid=os.getpid(),
        heartbeat_at=_stale_heartbeat_at(),
    )
    monkeypatch.setattr(
        "rl_fzerox.apps.run_manager.launching.save_games.career_mode_process_matches",
        lambda **_kwargs: True,
    )

    assert (
        active_career_mode_runner_pid(
            store=store,
            lease_id=lease_id,
            save_game_id=save_game.id,
        )
        is None
    )
    assert store.get_viewer_lease(lease_id) is None


def _stale_heartbeat_at() -> str:
    return (datetime.now(UTC) - timedelta(minutes=1)).isoformat(timespec="seconds")


def _configure_gp_cup(
    store: ManagerStore,
    *,
    save_game_id: str,
    run_id: str,
    cup_id: str,
) -> None:
    store.upsert_save_cup_setup(
        save_game_id=save_game_id,
        cup_id=cup_id,
        vehicle_id="blue_falcon",
    )
    for course in sorted(BUILT_IN_COURSES, key=lambda item: item.course_index):
        if course.cup != cup_id:
            continue
        store.upsert_save_course_setup(
            save_game_id=save_game_id,
            cup_id=cup_id,
            course_id=course.id,
            policy_run_id=run_id,
            policy_artifact="best",
        )
