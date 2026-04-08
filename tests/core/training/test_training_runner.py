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
    _atomic_save_artifact,
    _info_sequence,
    _resolve_policy_activation_fn,
    _resolve_train_run_config,
    _RolloutInfoAccumulator,
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


def test_rollout_info_accumulator_summarizes_state_and_episode_metrics() -> None:
    accumulator = _RolloutInfoAccumulator()
    infos = [
        {
            "race_distance": 10.0,
            "speed_kph": 100.0,
            "position": 5,
            "lap": 1,
            "laps_completed": 0,
            "episode": {
                "position": 2,
                "laps_completed": 3,
                "termination_reason": "finished",
                "truncation_reason": None,
            },
        },
        {
            "race_distance": 14.0,
            "speed_kph": 120.0,
            "position": 7,
            "lap": 1,
            "laps_completed": 0,
            "episode": {
                "position": 8,
                "laps_completed": 1,
                "termination_reason": None,
                "truncation_reason": "wrong_way",
            },
        },
    ]

    accumulator.add_infos(infos)

    assert accumulator.state_metrics["race_distance"].mean() == 12.0
    assert accumulator.state_metrics["speed_kph"].mean() == 110.0
    assert accumulator.episode_metrics["position"].mean() == 5.0
    assert accumulator.episode_metrics["laps_completed"].mean() == 2.0
    assert accumulator.episode_count == 2
    assert accumulator.termination_counts["finished"] == 1
    assert accumulator.truncation_counts["wrong_way"] == 1


def test_info_sequence_accepts_tuple_infos() -> None:
    infos = ({"race_distance": 10.0}, {"race_distance": 12.0})

    assert _info_sequence(infos) == infos
    assert _info_sequence([{"race_distance": 10.0}]) == [{"race_distance": 10.0}]
    assert _info_sequence(None) is None


def test_resolve_policy_activation_fn_supports_known_names() -> None:
    from torch import nn

    assert _resolve_policy_activation_fn("tanh") is nn.Tanh
    assert _resolve_policy_activation_fn("relu") is nn.ReLU


def test_resolve_policy_activation_fn_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unsupported policy activation"):
        _resolve_policy_activation_fn("gelu")


def test_atomic_save_artifact_replaces_target_without_leaving_tmp(tmp_path: Path) -> None:
    target_path = tmp_path / "latest_policy.zip"

    def _fake_save(path: str) -> None:
        Path(path).write_bytes(b"new-policy")

    _atomic_save_artifact(_fake_save, target_path)

    assert target_path.read_bytes() == b"new-policy"
    assert list(tmp_path.glob("*.tmp.zip")) == []
