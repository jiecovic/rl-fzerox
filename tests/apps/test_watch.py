# tests/apps/test_watch.py
from __future__ import annotations

from pathlib import Path

import pytest

from rl_fzerox.apps.watch import main
from rl_fzerox.core.config.schema import (
    EmulatorConfig,
    EnvConfig,
    PolicyConfig,
    TrainAppConfig,
    TrainConfig,
    WatchAppConfig,
    WatchConfig,
)


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
        policy=PolicyConfig(),
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
    assert config.watch.policy_run_dir == run_dir.resolve()
    assert config.watch.policy_artifact == "latest"
