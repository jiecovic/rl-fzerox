# tests/core/training/test_training_runner_config.py
from __future__ import annotations

from pathlib import Path

import pytest

from rl_fzerox.core.runtime_spec.schema import (
    EmulatorConfig,
    EnvConfig,
    PolicyConfig,
    TrackSamplingConfig,
    TrackSamplingEntryConfig,
    TrainAppConfig,
    TrainConfig,
)
from rl_fzerox.core.training import runner
from rl_fzerox.core.training.runs import build_run_paths
from rl_fzerox.core.training.session.artifacts import (
    resolve_train_run_config,
    validate_training_baseline_state,
)


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
        validate_training_baseline_state(config)


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

    validate_training_baseline_state(config)


def test_validate_training_baseline_state_checks_track_sampling_entries(
    tmp_path: Path,
) -> None:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    missing_baseline_path = tmp_path / "missing.state"
    core_path.touch()
    rom_path.touch()

    config = TrainAppConfig(
        seed=123,
        emulator=EmulatorConfig(core_path=core_path, rom_path=rom_path),
        env=EnvConfig(
            track_sampling=TrackSamplingConfig(
                enabled=True,
                entries=(
                    TrackSamplingEntryConfig(
                        id="missing",
                        baseline_state_path=missing_baseline_path,
                    ),
                ),
            ),
        ),
        policy=PolicyConfig(),
        train=TrainConfig(output_root=tmp_path / "runs", run_name="ppo_cnn"),
    )

    with pytest.raises(RuntimeError, match="missing.state"):
        validate_training_baseline_state(config)


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

    resolved_config = resolve_train_run_config(config=config, run_paths=run_paths)

    assert resolved_config.emulator.runtime_dir == run_paths.runtime_root


def test_run_training_removes_empty_run_dir_when_config_resolution_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    core_path.touch()
    rom_path.touch()
    config = TrainAppConfig(
        seed=123,
        emulator=EmulatorConfig(core_path=core_path, rom_path=rom_path),
        env=EnvConfig(),
        policy=PolicyConfig(),
        train=TrainConfig(output_root=tmp_path / "runs", run_name="cleanup_probe"),
    )

    def fail_config_resolution(**_: object) -> TrainAppConfig:
        raise RuntimeError("materialization failed")

    monkeypatch.setattr(runner, "resolve_train_run_config", fail_config_resolution)

    with pytest.raises(RuntimeError, match="materialization failed"):
        runner.run_training(config)

    assert not (tmp_path / "runs" / "cleanup_probe_0001").exists()


def test_run_training_keeps_existing_run_dir_when_in_place_continue_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    run_dir = tmp_path / "runs" / "cleanup_probe_0001"
    core_path.touch()
    rom_path.touch()
    run_dir.mkdir(parents=True)
    (run_dir / "keep.txt").write_text("keep", encoding="utf-8")
    config = TrainAppConfig(
        seed=123,
        emulator=EmulatorConfig(core_path=core_path, rom_path=rom_path),
        env=EnvConfig(),
        policy=PolicyConfig(),
        train=TrainConfig(
            output_root=tmp_path / "runs",
            run_name="cleanup_probe",
            continue_run_dir=run_dir,
            resume_run_dir=run_dir,
            resume_mode="full_model",
        ),
    )

    def fail_config_resolution(**_: object) -> TrainAppConfig:
        raise RuntimeError("materialization failed")

    monkeypatch.setattr(runner, "resolve_train_run_config", fail_config_resolution)

    with pytest.raises(RuntimeError, match="materialization failed"):
        runner.run_training(config)

    assert run_dir.exists()
    assert (run_dir / "keep.txt").read_text(encoding="utf-8") == "keep"
