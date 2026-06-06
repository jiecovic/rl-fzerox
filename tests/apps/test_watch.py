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
    ActionMaskConfig,
    CurriculumConfig,
    CurriculumStageConfig,
    CurriculumTriggerConfig,
    EmulatorConfig,
    EnvConfig,
    PolicyConfig,
    RewardConfig,
    TrainAppConfig,
    TrainConfig,
    WatchAppConfig,
    WatchConfig,
)
from rl_fzerox.core.training.inference import LoadedPolicy, PolicyRunner
from rl_fzerox.ui.watch.app import run_viewer
from rl_fzerox.ui.watch.runtime import WatchWorker
from rl_fzerox.ui.watch.runtime.policy import (
    _load_policy_runner,
    _policy_experience_frames,
    _sync_policy_curriculum_stage,
)


class _FakePolicyRunner:
    def __init__(self, stage_index: int | None) -> None:
        self.checkpoint_curriculum_stage_index = stage_index
        self.refresh_calls = 0
        self.reset_calls = 0

    @property
    def supports_action_masks(self) -> bool:
        return True

    def refresh(self) -> None:
        self.refresh_calls += 1

    def refresh_if_due(self, *, interval_seconds: float) -> None:
        del interval_seconds
        self.refresh()

    def reset(self) -> None:
        self.reset_calls += 1


class _FakeInferencePolicy:
    def predict(self, observation: object, **_kwargs: object) -> tuple[int, None]:
        del observation
        return 0, None


class _FakeWatchEnv:
    def __init__(self) -> None:
        self.stage_indices: list[int | None] = []

    def sync_checkpoint_curriculum_stage(self, stage_index: int | None) -> None:
        self.stage_indices.append(stage_index)


def test_watch_rejects_artifact_without_run_dir() -> None:
    with pytest.raises(SystemExit, match="--artifact requires --run-dir or --managed-run-id"):
        main(["--artifact", "best"])


def test_watch_rejects_missing_run_locator() -> None:
    with pytest.raises(
        SystemExit,
        match="--run-dir or --managed-run-id is required",
    ):
        main([])


def test_watch_clears_owned_viewer_lease_on_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    core_path = tmp_path / "core.so"
    rom_path = tmp_path / "rom.n64"
    run_dir = tmp_path / "runs" / "ppo_cnn_0001"
    core_path.touch()
    rom_path.touch()
    run_dir.mkdir(parents=True)
    lease_id = store.viewer_lease_id(
        kind="run_watch",
        owner_id="run-a",
        qualifier="latest",
    )
    store.upsert_viewer_lease(
        lease_id=lease_id,
        kind="run_watch",
        owner_id="run-a",
        pid=os.getpid(),
        qualifier="latest",
    )
    train_config = TrainAppConfig(
        seed=7,
        emulator=EmulatorConfig(
            core_path=core_path,
            rom_path=rom_path,
        ),
        env=EnvConfig(),
        policy=PolicyConfig(),
        train=TrainConfig(output_root=tmp_path / "runs", run_name="ppo_cnn"),
    )

    monkeypatch.setattr(
        "rl_fzerox.apps.watch_cli.resolve.load_train_run_config_for_watch",
        lambda *_args, **_kwargs: train_config,
    )
    monkeypatch.setattr(
        "rl_fzerox.apps.watch_cli.resolve.materialize_watch_session_config",
        lambda config, *, run_dir: config,
    )
    monkeypatch.setattr("rl_fzerox.apps.watch.run_viewer", lambda _config: None)

    main(
        [
            "--run-dir",
            str(run_dir),
            "--manager-db-path",
            str(store.db_path),
            "--viewer-lease-id",
            lease_id,
        ]
    )

    assert store.get_viewer_lease(lease_id) is None


def test_watch_allows_run_dir_overrides_without_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    core_path = tmp_path / "core.so"
    rom_path = tmp_path / "rom.n64"
    run_dir = tmp_path / "runs" / "ppo_cnn_0001"
    core_path.touch()
    rom_path.touch()
    run_dir.mkdir(parents=True)
    train_config = TrainAppConfig(
        seed=7,
        emulator=EmulatorConfig(
            core_path=core_path,
            rom_path=rom_path,
        ),
        env=EnvConfig(),
        policy=PolicyConfig(),
        train=TrainConfig(output_root=tmp_path / "runs", run_name="ppo_cnn"),
    )

    captured: dict[str, WatchAppConfig] = {}

    monkeypatch.setattr(
        "rl_fzerox.apps.watch_cli.resolve.load_train_run_config_for_watch",
        lambda *_args, **_kwargs: train_config,
    )
    monkeypatch.setattr(
        "rl_fzerox.apps.watch_cli.resolve.materialize_watch_session_config",
        lambda config, *, run_dir: config,
    )
    monkeypatch.setattr(
        "rl_fzerox.apps.watch.run_viewer",
        lambda config: captured.setdefault("config", config),
    )

    main(
        [
            "--run-dir",
            str(run_dir),
            "--",
            "watch.deterministic_policy=false",
            "watch.control_fps=30",
            "watch.render_fps=30",
        ]
    )

    config = captured["config"]
    assert config.watch.policy_run_dir == run_dir.resolve()
    assert config.watch.deterministic_policy is False
    assert config.watch.control_fps == 30.0
    assert config.watch.render_fps == 30.0


def test_resolve_watch_app_config_can_be_reused_by_headless_apps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    core_path = tmp_path / "core.so"
    rom_path = tmp_path / "rom.n64"
    run_dir = tmp_path / "runs" / "ppo_cnn_0001"
    core_path.touch()
    rom_path.touch()
    run_dir.mkdir(parents=True)
    train_config = TrainAppConfig(
        seed=7,
        emulator=EmulatorConfig(
            core_path=core_path,
            rom_path=rom_path,
        ),
        env=EnvConfig(),
        policy=PolicyConfig(),
        train=TrainConfig(output_root=tmp_path / "runs", run_name="ppo_cnn"),
    )

    monkeypatch.setattr(
        "rl_fzerox.apps.watch_cli.resolve.load_train_run_config_for_watch",
        lambda *_args, **_kwargs: train_config,
    )
    monkeypatch.setattr(
        "rl_fzerox.apps.watch_cli.resolve.materialize_watch_session_config",
        lambda config, **_kwargs: config,
    )

    config = resolve_watch_app_config(
        policy_run_dir=run_dir,
        policy_artifact="best",
        manager_db_path=None,
        managed_run_id=None,
        overrides=[],
    )

    assert config.watch.policy_run_dir == run_dir.resolve()
    assert config.watch.policy_artifact == "best"
    assert config.emulator.core_path == core_path
    assert config.train == train_config.train
    assert config.policy == train_config.policy


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
        "rl_fzerox.apps.watch_cli.resolve.materialize_train_run_config",
        lambda config, *, run_paths: config,
    )
    monkeypatch.setattr(
        "rl_fzerox.apps.watch_cli.resolve.materialize_watch_session_config",
        lambda config, *, run_dir: config,
    )

    config = resolve_watch_app_config(
        policy_run_dir=None,
        policy_artifact="latest",
        manager_db_path=db_path,
        managed_run_id=child.id,
        overrides=[],
    )

    assert config.env.action_repeat == 1
    assert config.watch.lineage_frame_offset == 2_000
    assert config.watch.manager_db_path == db_path.resolve()
    assert config.watch.managed_run_id == child.id


def test_watch_allows_run_dir_without_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    core_path = tmp_path / "core.so"
    rom_path = tmp_path / "rom.n64"
    run_dir = tmp_path / "runs" / "ppo_cnn_0001"
    core_path.touch()
    rom_path.touch()
    run_dir.mkdir(parents=True)

    train_config = TrainAppConfig(
        seed=7,
        emulator=EmulatorConfig(
            core_path=core_path,
            rom_path=rom_path,
        ),
        env=EnvConfig(action_repeat=2),
        reward=RewardConfig(time_penalty_per_frame=-0.123),
        policy=PolicyConfig(),
        curriculum=CurriculumConfig(
            enabled=True,
            stages=(
                CurriculumStageConfig(
                    name="basic_drive",
                    until=CurriculumTriggerConfig(race_laps_completed_mean_gte=3.0),
                    action_mask=ActionMaskConfig(lean=(0,)),
                ),
                CurriculumStageConfig(
                    name="lean_enabled",
                    action_mask=ActionMaskConfig(lean=(0, 1, 2)),
                ),
            ),
        ),
        train=TrainConfig(output_root=tmp_path / "runs", run_name="ppo_cnn"),
    )

    captured: dict[str, WatchAppConfig] = {}

    monkeypatch.setattr(
        "rl_fzerox.apps.watch_cli.resolve.load_train_run_config_for_watch",
        lambda *_args, **_kwargs: train_config,
    )
    monkeypatch.setattr(
        "rl_fzerox.apps.watch_cli.resolve.materialize_watch_session_config",
        lambda config, *, run_dir: config,
    )
    monkeypatch.setattr(
        "rl_fzerox.apps.watch.run_viewer",
        lambda config: captured.setdefault("config", config),
    )

    main(["--run-dir", str(run_dir)])

    config = captured["config"]
    assert config.seed == 7
    assert config.emulator.core_path == core_path.resolve()
    assert config.emulator.rom_path == rom_path.resolve()
    assert config.env.action_repeat == 2
    assert config.reward.time_penalty_per_frame == -0.123
    assert config.curriculum.enabled is True
    assert config.curriculum.stages[0].name == "basic_drive"
    assert config.train == train_config.train
    assert config.policy == train_config.policy
    assert config.watch.device == "cpu"
    assert config.watch.policy_run_dir == run_dir.resolve()
    assert config.watch.policy_artifact == "latest"


def test_watch_cli_overrides_apply_after_run_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    core_path = tmp_path / "core.so"
    rom_path = tmp_path / "rom.n64"
    run_dir = tmp_path / "runs" / "ppo_cnn_0001"
    core_path.touch()
    rom_path.touch()
    run_dir.mkdir(parents=True)

    train_config = TrainAppConfig(
        seed=7,
        emulator=EmulatorConfig(
            core_path=core_path,
            rom_path=rom_path,
        ),
        env=EnvConfig(action_repeat=3, camera_setting="regular"),
        policy=PolicyConfig(),
        train=TrainConfig(output_root=tmp_path / "runs", run_name="ppo_cnn"),
    )

    captured: dict[str, WatchAppConfig] = {}
    monkeypatch.setattr(
        "rl_fzerox.apps.watch_cli.resolve.load_train_run_config_for_watch",
        lambda *_args, **_kwargs: train_config,
    )
    monkeypatch.setattr(
        "rl_fzerox.apps.watch_cli.resolve.materialize_watch_session_config",
        lambda config, *, run_dir: config,
    )
    monkeypatch.setattr(
        "rl_fzerox.apps.watch.run_viewer",
        lambda config: captured.setdefault("config", config),
    )

    main(
        [
            "--run-dir",
            str(run_dir),
            "--",
            "env.camera_setting=close_behind",
            "watch.control_fps=15",
            "watch.render_fps=15",
        ]
    )

    config = captured["config"]
    assert config.seed == 7
    assert config.env.action_repeat == 3
    assert config.env.camera_setting == "close_behind"
    assert config.watch.control_fps == 15.0
    assert config.watch.render_fps == 15.0
    assert config.watch.policy_run_dir == run_dir.resolve()


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


def test_sync_policy_curriculum_stage_applies_checkpoint_stage_to_watch_env() -> None:
    policy_runner = _FakePolicyRunner(stage_index=0)
    env = _FakeWatchEnv()

    _sync_policy_curriculum_stage(policy_runner, env)

    assert policy_runner.refresh_calls == 1
    assert env.stage_indices == [0]


def test_sync_policy_curriculum_stage_resets_when_checkpoint_stage_is_missing() -> None:
    policy_runner = _FakePolicyRunner(stage_index=None)
    env = _FakeWatchEnv()

    _sync_policy_curriculum_stage(policy_runner, env)

    assert policy_runner.refresh_calls == 1
    assert env.stage_indices == [None]


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
        lambda _worker: (SimpleNamespace(native_fps=30.0), False),
    )
    monkeypatch.setattr("rl_fzerox.ui.watch.app._create_fonts", lambda _pygame: object())
    monkeypatch.setattr(
        "rl_fzerox.ui.watch.app._watch_game_display_size",
        lambda: (640, 480),
    )

    run_viewer(config, worker_factory=lambda _config: _FakeWorker())

    assert calls == ["init", "shutdown", "quit"]
