# tests/apps/test_watch.py
from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from rl_fzerox.apps.watch import main, resolve_watch_app_config
from rl_fzerox.core.manager import ManagerStore, default_managed_run_config
from rl_fzerox.core.runtime_spec.schema import (
    EmulatorConfig,
    WatchAppConfig,
    WatchConfig,
)
from rl_fzerox.core.training.inference import LoadedPolicy, PolicyRunner
from rl_fzerox.ui.watch.app import run_viewer
from rl_fzerox.ui.watch.runtime import WatchWorker
from rl_fzerox.ui.watch.runtime.policy import (
    _load_policy_runner,
    _policy_experience_frames,
)


class _FakeInferencePolicy:
    def predict(self, observation: object, **_kwargs: object) -> tuple[int, None]:
        del observation
        return 0, None


def test_watch_rejects_artifact_without_run_id() -> None:
    with pytest.raises(SystemExit, match="--run-id is required"):
        main(["--artifact", "best"])


def test_watch_rejects_missing_run_id() -> None:
    with pytest.raises(SystemExit, match="--run-id is required"):
        main([])


def test_watch_clears_owned_viewer_lease_on_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    run = store.create_run(
        run_id="run-a",
        name="Run A",
        config=default_managed_run_config(),
        managed_runs_root=tmp_path / "runs",
    )
    run.run_dir.mkdir(parents=True)
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
    captured_materialize_kwargs: dict[str, object] = {}

    monkeypatch.setattr(
        "rl_fzerox.core.manager.projection.watch.materialize_train_run_config",
        lambda config, *, run_paths: config,
    )
    monkeypatch.setattr(
        "rl_fzerox.core.manager.projection.watch.materialize_watch_session_config",
        lambda config, **kwargs: (captured_materialize_kwargs.update(kwargs) or config),
    )
    monkeypatch.setattr("rl_fzerox.apps.watch.run_viewer", lambda _config, **_kwargs: None)

    main(
        [
            "--run-id",
            run.id,
            "--manager-db-path",
            str(store.db_path),
            "--viewer-lease-id",
            lease_id,
        ]
    )

    assert store.get_viewer_lease(lease_id) is None
    assert captured_materialize_kwargs["session_name"] == f"{lease_id}:{os.getpid()}"


def test_watch_allows_run_id_overrides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    run = store.create_run(
        run_id="watch-overrides",
        name="Watch overrides",
        config=default_managed_run_config(),
        managed_runs_root=tmp_path / "runs",
    )
    run.run_dir.mkdir(parents=True)

    captured: dict[str, WatchAppConfig] = {}

    monkeypatch.setattr(
        "rl_fzerox.core.manager.projection.watch.materialize_train_run_config",
        lambda config, *, run_paths: config,
    )
    monkeypatch.setattr(
        "rl_fzerox.core.manager.projection.watch.materialize_watch_session_config",
        lambda config, **_kwargs: config,
    )
    monkeypatch.setattr(
        "rl_fzerox.apps.watch.run_viewer",
        lambda config, **_kwargs: captured.setdefault("config", config),
    )

    main(
        [
            "--run-id",
            run.id,
            "--manager-db-path",
            str(store.db_path),
            "--",
            "watch.deterministic_policy=false",
            "watch.control_fps=30",
            "watch.render_fps=30",
        ]
    )

    config = captured["config"]
    assert config.watch.policy_run_dir == run.run_dir.resolve()
    assert config.watch.managed_run_id == run.id
    assert config.watch.deterministic_policy is False
    assert config.watch.control_fps == 30.0
    assert config.watch.render_fps == 30.0


def test_resolve_watch_app_config_can_be_reused_by_headless_apps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    run = store.create_run(
        run_id="headless-watch",
        name="Headless watch",
        config=default_managed_run_config(),
        managed_runs_root=tmp_path / "runs",
    )
    run.run_dir.mkdir(parents=True)

    monkeypatch.setattr(
        "rl_fzerox.core.manager.projection.watch.materialize_train_run_config",
        lambda config, *, run_paths: config,
    )
    monkeypatch.setattr(
        "rl_fzerox.core.manager.projection.watch.materialize_watch_session_config",
        lambda config, **_kwargs: config,
    )

    config = resolve_watch_app_config(
        run_id=run.id,
        policy_artifact="best",
        manager_db_path=store.db_path,
        overrides=[],
    )

    assert config.watch.policy_run_dir == run.run_dir.resolve()
    assert config.watch.policy_artifact == "best"
    assert config.watch.manager_db_path == store.db_path.resolve()
    assert config.watch.managed_run_id == run.id
    assert config.train is not None
    assert config.policy is not None


def test_resolve_watch_app_config_syncs_manifest_mirror_from_sqlite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "manager" / "runs.db"
    store = ManagerStore(db_path)
    run = store.create_run(
        run_id="manifest-watch",
        name="Manifest Watch",
        config=default_managed_run_config(),
        explicit_run_dir=tmp_path / "runs" / "manifest-watch",
    )
    run.run_dir.mkdir(parents=True)
    (run.run_dir / "train_manifest.yaml").write_text(
        "removed_top_level_field: stale\n"
        "env:\n"
        "  action_repeat: 99\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "rl_fzerox.core.manager.projection.watch.materialize_train_run_config",
        lambda config, *, run_paths: config,
    )
    monkeypatch.setattr(
        "rl_fzerox.core.manager.projection.watch.materialize_watch_session_config",
        lambda config, **_kwargs: config,
    )

    config = resolve_watch_app_config(
        run_id=run.id,
        policy_artifact="latest",
        manager_db_path=db_path,
        overrides=[],
    )

    assert config.watch.policy_run_dir == run.run_dir.resolve()
    assert config.watch.manager_db_path == db_path.resolve()
    assert config.watch.managed_run_id == run.id
    assert config.env.action_repeat == default_managed_run_config().action.action_repeat
    manifest_text = (run.run_dir / "train_manifest.yaml").read_text(encoding="utf-8")
    assert "removed_top_level_field" not in manifest_text
    assert "action_repeat: 99" not in manifest_text


def test_resolve_watch_app_config_tracks_managed_lineage_frame_offset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "manager" / "runs.db"
    store = ManagerStore(db_path)
    parent_config = default_managed_run_config().model_copy(deep=True)
    parent_config.action.action_repeat = 2
    child_config = default_managed_run_config().model_copy(deep=True)
    child_config.action.action_repeat = 1
    parent = store.create_run(
        run_id="parent",
        name="Parent",
        config=parent_config,
        managed_runs_root=tmp_path / "runs",
    )
    child = store.create_run(
        run_id="child",
        name="Child",
        config=child_config,
        managed_runs_root=tmp_path / "runs",
        lineage_id=parent.lineage_id,
        lineage_step_offset=1_000,
        parent_run_id=parent.id,
        source_run_id=parent.id,
        source_artifact="latest",
        source_num_timesteps=1_000,
    )
    child.run_dir.mkdir(parents=True)

    monkeypatch.setattr(
        "rl_fzerox.core.manager.projection.watch.materialize_train_run_config",
        lambda config, *, run_paths: config,
    )
    monkeypatch.setattr(
        "rl_fzerox.core.manager.projection.watch.materialize_watch_session_config",
        lambda config, **_kwargs: config,
    )

    config = resolve_watch_app_config(
        run_id=child.id,
        policy_artifact="latest",
        manager_db_path=db_path,
        overrides=[],
    )

    assert config.env.action_repeat == 1
    assert config.watch.lineage_frame_offset == 2_000
    assert config.watch.manager_db_path == db_path.resolve()
    assert config.watch.managed_run_id == child.id


def test_load_policy_runner_uses_configured_watch_device(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def _fake_load_policy_runner(
        run_dir: Path,
        *,
        artifact: str,
        device: str,
        algorithm: str | None = None,
    ) -> str:
        captured["run_dir"] = run_dir
        captured["artifact"] = artifact
        captured["device"] = device
        captured["algorithm"] = algorithm
        return "runner"

    monkeypatch.setattr(
        "rl_fzerox.core.training.inference.load_policy_runner",
        _fake_load_policy_runner,
    )

    run_dir = Path("/tmp/example-run")
    assert _load_policy_runner(run_dir, artifact="best", device="cpu") == "runner"
    assert captured == {
        "run_dir": run_dir,
        "artifact": "best",
        "device": "cpu",
        "algorithm": None,
    }


def test_policy_experience_frames_prefers_managed_frame_offset(tmp_path: Path) -> None:
    policy_path = tmp_path / "policy.zip"
    policy_path.touch()
    runner = PolicyRunner(
        LoadedPolicy(
            run_dir=tmp_path,
            policy_path=policy_path,
            artifact="latest",
            num_timesteps=1_000,
            lineage_num_timesteps=5_000,
        ),
        _FakeInferencePolicy(),
    )

    assert (
        _policy_experience_frames(
            runner,
            action_repeat=1,
            lineage_frame_offset=8_000,
        )
        == 9_000
    )
    assert (
        _policy_experience_frames(
            runner,
            action_repeat=2,
            lineage_frame_offset=None,
        )
        == 10_000
    )


def test_run_viewer_exits_quietly_on_keyboard_interrupt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    core_path = tmp_path / "core.so"
    rom_path = tmp_path / "rom.n64"
    core_path.touch()
    rom_path.touch()

    config = WatchAppConfig(
        seed=7,
        emulator=EmulatorConfig(
            core_path=core_path,
            rom_path=rom_path,
        ),
        watch=WatchConfig(),
    )

    calls: list[str] = []

    class _FakeClock:
        def tick(self, _limit: int) -> None:
            raise KeyboardInterrupt

    fake_pygame = SimpleNamespace(
        init=lambda: calls.append("init"),
        quit=lambda: calls.append("quit"),
        time=SimpleNamespace(Clock=lambda: _FakeClock()),
    )

    class _FakeWorker(WatchWorker):
        def __init__(self) -> None:
            pass

        def shutdown(self) -> None:
            calls.append("shutdown")

    monkeypatch.setitem(sys.modules, "pygame", fake_pygame)
    monkeypatch.setattr("rl_fzerox.ui.watch.app.start_watch_worker", lambda _config: _FakeWorker())
    monkeypatch.setattr(
        "rl_fzerox.ui.watch.app.AuxiliaryEpisodeMetricsTracker.observe_snapshot",
        lambda _self, _snapshot: None,
    )
    monkeypatch.setattr(
        "rl_fzerox.ui.watch.app.wait_initial_snapshot",
        lambda _worker, **_kwargs: (
            SimpleNamespace(
                native_fps=30.0,
                active_track_sampling=None,
                info={},
                policy_observation_shape=None,
            ),
            False,
        ),
    )
    monkeypatch.setattr("rl_fzerox.ui.watch.app._create_fonts", lambda _pygame: object())
    monkeypatch.setattr(
        "rl_fzerox.ui.watch.app._watch_game_display_size",
        lambda: (640, 480),
    )

    run_viewer(config, worker_factory=lambda _config: _FakeWorker())

    assert calls == ["init", "shutdown", "quit"]
