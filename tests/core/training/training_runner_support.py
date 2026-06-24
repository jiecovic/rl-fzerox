# tests/core/training/training_runner_support.py
from __future__ import annotations

from pathlib import Path

import pytest

from rl_fzerox.core.runtime_spec.schema import (
    EmulatorConfig,
    EnvConfig,
    PolicyConfig,
    TrainAppConfig,
    TrainConfig,
)
from rl_fzerox.core.training import runner


class _CapturingLogger:
    def __init__(self) -> None:
        self.records: dict[str, object] = {}

    def record(self, key: str, value: object) -> None:
        self.records[key] = value


class _TrainingStepModel:
    def __init__(self, num_timesteps: object) -> None:
        self.num_timesteps = num_timesteps


class _RunTrainingEnv:
    def __init__(self) -> None:
        self.closed = False
        self.env_method_calls: list[tuple[str, tuple[object, ...]]] = []

    def env_method(self, method_name: str, *args: object) -> None:
        self.env_method_calls.append((method_name, args))

    def close(self) -> None:
        self.closed = True


class _RunTrainingModel:
    def __init__(self, num_timesteps: int) -> None:
        self.num_timesteps = num_timesteps
        self.logger: object | None = None

    def set_logger(self, logger: object) -> None:
        self.logger = logger


def _full_model_resume_config(tmp_path: Path, *, total_timesteps: int) -> TrainAppConfig:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    run_dir = tmp_path / "runs" / "ppo_cnn_0001"
    core_path.touch()
    rom_path.touch()
    run_dir.mkdir(parents=True)
    return TrainAppConfig(
        emulator=EmulatorConfig(core_path=core_path, rom_path=rom_path),
        env=EnvConfig(),
        policy=PolicyConfig(),
        train=TrainConfig(
            continue_run_dir=run_dir,
            resume_run_dir=run_dir,
            resume_mode="full_model",
            save_latest_checkpoint=False,
            total_timesteps=total_timesteps,
        ),
    )


def _stub_run_training_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    *,
    config: TrainAppConfig,
    env: _RunTrainingEnv,
    model: _RunTrainingModel,
) -> None:
    monkeypatch.setattr(runner, "resolve_train_run_config", lambda **_: config)
    monkeypatch.setattr(runner, "build_training_env", lambda *_, **__: env)
    monkeypatch.setattr(runner, "build_training_model", lambda **_: model)
    monkeypatch.setattr(runner, "maybe_resume_training_model", lambda **_: model)
    monkeypatch.setattr(runner, "build_callbacks", lambda **_: object())
    monkeypatch.setattr(runner, "build_tensorboard_logger", lambda *_, **__: object())
    monkeypatch.setattr(runner, "print_training_startup", lambda **_: None)
    monkeypatch.setattr(runner, "current_policy_artifact_metadata", lambda *_, **__: None)
    monkeypatch.setattr(runner, "save_artifacts_atomically", lambda **_: None)
