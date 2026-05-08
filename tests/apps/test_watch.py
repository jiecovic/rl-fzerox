# tests/apps/test_watch.py
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from rl_fzerox.apps.watch import main, resolve_watch_app_config
from rl_fzerox.core.runtime_spec.schema import (
    ActionMaskConfig,
    CurriculumConfig,
    CurriculumStageConfig,
    CurriculumTriggerConfig,
    EmulatorConfig,
    EnvConfig,
    PolicyConfig,
    RewardConfig,
    TrackSamplingConfig,
    TrackSamplingEntryConfig,
    TrainAppConfig,
    TrainConfig,
    WatchAppConfig,
    WatchConfig,
)
from rl_fzerox.ui.watch.app import run_viewer
from rl_fzerox.ui.watch.runtime.policy import _load_policy_runner, _sync_policy_curriculum_stage


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

    def reset(self) -> None:
        self.reset_calls += 1


class _FakeWatchEnv:
    def __init__(self) -> None:
        self.stage_indices: list[int | None] = []

    def sync_checkpoint_curriculum_stage(self, stage_index: int | None) -> None:
        self.stage_indices.append(stage_index)


def test_watch_rejects_artifact_without_run_dir() -> None:
    with pytest.raises(SystemExit, match="--artifact requires --run-dir or --managed-run-id"):
        main(["--artifact", "best"])


def test_watch_rejects_missing_run_locator() -> None:
    with pytest.raises(SystemExit, match="--run-dir or --managed-run-id is required"):
        main([])


def test_watch_removes_owned_pid_file_on_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    core_path = tmp_path / "core.so"
    rom_path = tmp_path / "rom.n64"
    run_dir = tmp_path / "runs" / "ppo_cnn_0001"
    core_path.touch()
    rom_path.touch()
    run_dir.mkdir(parents=True)
    pid_path = tmp_path / "watch.json"
    pid_path.write_text(json.dumps({"pid": os.getpid()}) + "\n", encoding="utf-8")
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

    main(["--run-dir", str(run_dir), "--watch-pid-file", str(pid_path)])

    assert not pid_path.exists()


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


def test_resolve_watch_app_config_disables_track_sampling_for_x_cup_watch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    core_path = tmp_path / "core.so"
    rom_path = tmp_path / "rom.n64"
    baseline_state_path = tmp_path / "track.state"
    run_dir = tmp_path / "runs" / "ppo_cnn_0001"
    core_path.touch()
    rom_path.touch()
    baseline_state_path.write_bytes(b"baseline")
    run_dir.mkdir(parents=True)
    train_config = TrainAppConfig(
        seed=7,
        emulator=EmulatorConfig(
            core_path=core_path,
            rom_path=rom_path,
        ),
        env=EnvConfig(
            track_sampling=TrackSamplingConfig(
                enabled=True,
                entries=(
                    TrackSamplingEntryConfig(
                        id="mute_city",
                        course_id="mute_city",
                        baseline_state_path=baseline_state_path,
                    ),
                ),
            )
        ),
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
        policy_artifact="latest",
        manager_db_path=None,
        managed_run_id=None,
        overrides=["watch.x_cup.enabled=true"],
    )

    assert config.watch.x_cup.enabled is True
    assert config.env.track_sampling.enabled is False


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

    class _FakeWorker:
        def shutdown(self) -> None:
            calls.append("shutdown")

    monkeypatch.setitem(sys.modules, "pygame", fake_pygame)
    monkeypatch.setattr("rl_fzerox.ui.watch.app.start_watch_worker", lambda _config: _FakeWorker())
    monkeypatch.setattr(
        "rl_fzerox.ui.watch.app.wait_initial_snapshot",
        lambda _worker: (SimpleNamespace(native_fps=30.0), False),
    )
    monkeypatch.setattr("rl_fzerox.ui.watch.app._create_fonts", lambda _pygame: object())
    monkeypatch.setattr(
        "rl_fzerox.ui.watch.app._watch_game_display_size",
        lambda: (640, 480),
    )

    run_viewer(config)

    assert calls == ["init", "shutdown", "quit"]
