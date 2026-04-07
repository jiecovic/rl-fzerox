# tests/core/training/test_training_runner.py
from __future__ import annotations

from pathlib import Path

import pytest

from rl_fzerox.core.config.schema import (
    EmulatorConfig,
    EnvConfig,
    PolicyConfig,
    TrainAppConfig,
    TrainConfig,
)
from rl_fzerox.core.training.runner import (
    _resolve_train_run_config,
    _validate_training_baseline_state,
)
from rl_fzerox.core.training.runs import build_run_paths


def test_validate_training_baseline_state_requires_existing_file(
    tmp_path: Path,
) -> None:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    baseline_state_path = tmp_path / "first-race.state"
    core_path.touch()
    rom_path.touch()

    config = TrainAppConfig(
        seed=123,
        emulator=EmulatorConfig(
            core_path=core_path,
            rom_path=rom_path,
            baseline_state_path=baseline_state_path,
        ),
        env=EnvConfig(),
        policy=PolicyConfig(),
        train=TrainConfig(output_root=tmp_path / "runs", run_name="ppo_cnn"),
    )

    with pytest.raises(RuntimeError, match="Configured training baseline state"):
        _validate_training_baseline_state(config)


def test_validate_training_baseline_state_accepts_existing_file(tmp_path: Path) -> None:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    baseline_state_path = tmp_path / "first-race.state"
    core_path.touch()
    rom_path.touch()
    baseline_state_path.write_bytes(b"baseline")

    config = TrainAppConfig(
        seed=123,
        emulator=EmulatorConfig(
            core_path=core_path,
            rom_path=rom_path,
            baseline_state_path=baseline_state_path,
        ),
        env=EnvConfig(),
        policy=PolicyConfig(),
        train=TrainConfig(output_root=tmp_path / "runs", run_name="ppo_cnn"),
    )

    _validate_training_baseline_state(config)


def test_resolve_train_run_config_sets_run_local_runtime_root(tmp_path: Path) -> None:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    core_path.touch()
    rom_path.touch()

    config = TrainAppConfig(
        seed=123,
        emulator=EmulatorConfig(core_path=core_path, rom_path=rom_path),
        env=EnvConfig(),
        policy=PolicyConfig(),
        train=TrainConfig(output_root=tmp_path / "runs", run_name="ppo_cnn"),
    )
    run_paths = build_run_paths(
        output_root=config.train.output_root,
        run_name=config.train.run_name,
    )

    resolved_config = _resolve_train_run_config(config=config, run_paths=run_paths)

    assert resolved_config.emulator.runtime_dir == run_paths.runtime_root
