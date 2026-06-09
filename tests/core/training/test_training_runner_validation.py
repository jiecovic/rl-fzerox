# tests/core/training/test_training_runner_validation.py
from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from rl_fzerox.core.runtime_spec.schema import (
    ActionConfig,
    ActionMaskConfig,
    CurriculumConfig,
    EmulatorConfig,
    EnvConfig,
    PolicyConfig,
    PolicyRecurrentConfig,
    TrainAppConfig,
    TrainConfig,
)
from rl_fzerox.core.training.runs import RUN_LAYOUT
from rl_fzerox.core.training.session.artifacts import (
    atomic_save_artifact,
)
from rl_fzerox.core.training.session.model import (
    resolve_policy_activation_fn,
)


def test_resolve_policy_activation_fn_supports_known_names() -> None:
    from torch import nn

    assert resolve_policy_activation_fn("tanh") is nn.Tanh
    assert resolve_policy_activation_fn("relu") is nn.ReLU
    assert resolve_policy_activation_fn("gelu") is nn.GELU


def test_resolve_policy_activation_fn_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unsupported policy activation"):
        resolve_policy_activation_fn("silu")


def test_atomic_save_artifact_replaces_target_without_leaving_tmp(tmp_path: Path) -> None:
    target_path = tmp_path / RUN_LAYOUT.policy_artifacts.latest

    def _fake_save(path: str) -> None:
        Path(path).write_bytes(b"new-policy")

    atomic_save_artifact(_fake_save, target_path)

    assert target_path.read_bytes() == b"new-policy"
    assert list(target_path.parent.glob("*.tmp.zip")) == []


def test_train_config_rejects_plain_ppo_algorithm(
    tmp_path: Path,
) -> None:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    core_path.touch()
    rom_path.touch()

    with pytest.raises(ValidationError, match="algorithm"):
        TrainAppConfig(
            emulator=EmulatorConfig(core_path=core_path, rom_path=rom_path),
            env=EnvConfig(action=ActionConfig(mask=ActionMaskConfig(lean=(0,)))),
            policy=PolicyConfig(),
            curriculum=CurriculumConfig(),
            train=TrainConfig.model_validate({"algorithm": "ppo"}),
        )


def test_train_app_config_rejects_recurrent_policy_without_recurrent_algorithm(
    tmp_path: Path,
) -> None:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    core_path.touch()
    rom_path.touch()

    with pytest.raises(
        ValidationError,
        match="policy.recurrent.enabled=true requires a recurrent train.algorithm",
    ):
        TrainAppConfig(
            emulator=EmulatorConfig(core_path=core_path, rom_path=rom_path),
            env=EnvConfig(),
            policy=PolicyConfig(
                recurrent=PolicyRecurrentConfig(enabled=True),
            ),
            curriculum=CurriculumConfig(),
            train=TrainConfig(algorithm="maskable_hybrid_action_ppo"),
        )


def test_train_app_config_rejects_recurrent_algorithm_without_recurrent_policy(
    tmp_path: Path,
) -> None:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    core_path.touch()
    rom_path.touch()

    with pytest.raises(
        ValidationError,
        match=(
            "train.algorithm=maskable_hybrid_recurrent_ppo requires policy.recurrent.enabled=true"
        ),
    ):
        TrainAppConfig(
            emulator=EmulatorConfig(core_path=core_path, rom_path=rom_path),
            env=EnvConfig(),
            policy=PolicyConfig(),
            curriculum=CurriculumConfig(),
            train=TrainConfig(algorithm="maskable_hybrid_recurrent_ppo"),
        )
