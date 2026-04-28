# tests/apps/test_train.py
from __future__ import annotations

from pathlib import Path

import pytest

from rl_fzerox.apps.train import main, parse_args
from rl_fzerox.core.config.schema import (
    EmulatorConfig,
    EnvConfig,
    PolicyConfig,
    TrainAppConfig,
    TrainConfig,
)


def test_parse_args_accepts_continue_run_dir_without_config() -> None:
    args = parse_args(["--continue-run-dir", "local/runs/probe_0001"])

    assert args.config_path is None
    assert args.continue_run_dir == Path("local/runs/probe_0001")
    assert args.continue_artifact == "latest"


def test_main_loads_saved_run_config_for_in_place_continue_without_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    core_path = tmp_path / "core.so"
    rom_path = tmp_path / "rom.n64"
    run_dir = tmp_path / "runs" / "ppo_cnn_0001"
    core_path.touch()
    rom_path.touch()
    run_dir.mkdir(parents=True)

    captured: dict[str, TrainAppConfig] = {}
    base_config = TrainAppConfig(
        emulator=EmulatorConfig(core_path=core_path, rom_path=rom_path),
        env=EnvConfig(),
        policy=PolicyConfig(),
        train=TrainConfig(
            output_root=tmp_path / "runs",
            run_name="ppo_cnn",
        ),
    )

    def _load_train_run_config(path: Path) -> TrainAppConfig:
        assert path == run_dir.resolve()
        return base_config

    monkeypatch.setattr(
        "rl_fzerox.apps.train.load_train_run_config",
        _load_train_run_config,
    )
    monkeypatch.setattr(
        "rl_fzerox.apps.train.run_training",
        lambda config: captured.setdefault("config", config),
    )

    main(
        [
            "--continue-run-dir",
            str(run_dir),
        ]
    )

    config = captured["config"]
    assert config.train.continue_run_dir == run_dir.resolve()
    assert config.train.resume_run_dir == run_dir.resolve()
    assert config.train.resume_artifact == "latest"
    assert config.train.resume_mode == "full_model"


def test_main_applies_overrides_to_saved_run_continue_without_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    core_path = tmp_path / "core.so"
    rom_path = tmp_path / "rom.n64"
    run_dir = tmp_path / "runs" / "ppo_cnn_0001"
    core_path.touch()
    rom_path.touch()
    run_dir.mkdir(parents=True)

    captured: dict[str, TrainAppConfig] = {}
    monkeypatch.setattr(
        "rl_fzerox.apps.train.load_train_run_config",
        lambda _path: TrainAppConfig(
            emulator=EmulatorConfig(core_path=core_path, rom_path=rom_path),
            env=EnvConfig(),
            policy=PolicyConfig(),
            train=TrainConfig(
                output_root=tmp_path / "runs",
                run_name="ppo_cnn",
            ),
        ),
    )
    monkeypatch.setattr(
        "rl_fzerox.apps.train.run_training",
        lambda config: captured.setdefault("config", config),
    )

    main(
        [
            "--continue-run-dir",
            str(run_dir),
            "--continue-artifact",
            "best",
            "--",
            "train.total_timesteps=2000",
            "train.num_envs=10",
        ]
    )

    config = captured["config"]
    assert config.train.continue_run_dir == run_dir.resolve()
    assert config.train.resume_run_dir == run_dir.resolve()
    assert config.train.resume_artifact == "best"
    assert config.train.resume_mode == "full_model"
    assert config.train.total_timesteps == 2000
    assert config.train.num_envs == 10
