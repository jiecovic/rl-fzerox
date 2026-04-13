# tests/apps/test_watch.py
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from rl_fzerox.apps.watch import main
from rl_fzerox.core.config.schema import (
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
from rl_fzerox.ui.watch.session import _load_policy_runner, _sync_policy_curriculum_stage


class _FakePolicyRunner:
    def __init__(self, stage_index: int | None) -> None:
        self.checkpoint_curriculum_stage_index = stage_index
        self.refresh_calls = 0
        self.reset_calls = 0

    def refresh(self) -> None:
        self.refresh_calls += 1

    def reset(self) -> None:
        self.reset_calls += 1


class _FakeWatchEnv:
    def __init__(self) -> None:
        self.stage_indices: list[int | None] = []

    def sync_checkpoint_curriculum_stage(self, stage_index: int | None) -> None:
        self.stage_indices.append(stage_index)


def test_watch_rejects_artifact_without_run_dir(
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

    monkeypatch.setattr(
        "rl_fzerox.apps.watch.load_watch_app_config",
        lambda *args, **kwargs: config,
    )

    with pytest.raises(
        SystemExit,
        match="--artifact requires --run-dir or watch.policy_run_dir",
    ):
        main(["--config", str(tmp_path / "watch.yaml"), "--artifact", "best"])


def test_watch_rejects_missing_config_without_run_dir() -> None:
    with pytest.raises(SystemExit, match="--config is required unless --run-dir is provided"):
        main([])


def test_watch_rejects_overrides_without_config(
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
        "rl_fzerox.apps.watch.load_train_run_config",
        lambda *_args, **_kwargs: train_config,
    )

    with pytest.raises(SystemExit, match="Hydra overrides require --config"):
        main(["--run-dir", str(run_dir), "--", "watch.fps=30"])


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
                    action_mask=ActionMaskConfig(shoulder=(0,)),
                ),
                CurriculumStageConfig(
                    name="drift_enabled",
                    action_mask=ActionMaskConfig(shoulder=(0, 1, 2)),
                ),
            ),
        ),
        train=TrainConfig(output_root=tmp_path / "runs", run_name="ppo_cnn"),
    )

    captured: dict[str, WatchAppConfig] = {}

    monkeypatch.setattr(
        "rl_fzerox.apps.watch.load_train_run_config",
        lambda *_args, **_kwargs: train_config,
    )
    monkeypatch.setattr(
        "rl_fzerox.apps.watch.materialize_watch_session_config",
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

    watch_config = WatchAppConfig(
        seed=999,
        emulator=EmulatorConfig(
            core_path=core_path,
            rom_path=rom_path,
        ),
        env=EnvConfig(action_repeat=1, camera_setting="close_behind"),
        watch=WatchConfig(fps=30.0),
    )
    overridden_watch_config = watch_config.model_copy(
        update={
            "env": watch_config.env.model_copy(update={"camera_setting": "close_behind"}),
            "watch": watch_config.watch.model_copy(update={"fps": 15.0}),
        }
    )
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

    def _load_watch_config(
        _config_path: Path,
        *,
        overrides: Sequence[str] | None = None,
    ) -> WatchAppConfig:
        return overridden_watch_config if overrides else watch_config

    captured: dict[str, WatchAppConfig] = {}

    monkeypatch.setattr("rl_fzerox.apps.watch.load_watch_app_config", _load_watch_config)
    monkeypatch.setattr(
        "rl_fzerox.apps.watch.load_train_run_config",
        lambda *_args, **_kwargs: train_config,
    )
    monkeypatch.setattr(
        "rl_fzerox.apps.watch.materialize_watch_session_config",
        lambda config, *, run_dir: config,
    )
    monkeypatch.setattr(
        "rl_fzerox.apps.watch.run_viewer",
        lambda config: captured.setdefault("config", config),
    )

    main(
        [
            "--config",
            str(tmp_path / "watch.yaml"),
            "--run-dir",
            str(run_dir),
            "--",
            "env.camera_setting=close_behind",
            "watch.fps=15",
        ]
    )

    config = captured["config"]
    assert config.seed == 7
    assert config.env.action_repeat == 3
    assert config.env.camera_setting == "close_behind"
    assert config.watch.fps == 15.0
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
    ) -> str:
        captured["run_dir"] = run_dir
        captured["artifact"] = artifact
        captured["device"] = device
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
