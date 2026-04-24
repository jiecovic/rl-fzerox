# tests/core/training/test_training_algorithms.py
from __future__ import annotations

from pathlib import Path

import pytest

from rl_fzerox.core.config.schema import (
    ActionConfig,
    ActionMaskConfig,
    CurriculumConfig,
    EmulatorConfig,
    EnvConfig,
    ObservationConfig,
    PolicyConfig,
    PolicyRecurrentConfig,
    TrainAppConfig,
    TrainConfig,
)
from rl_fzerox.core.envs import FZeroXEnv
from rl_fzerox.core.training.session.model import (
    build_ppo_model,
    build_training_model,
    resolve_effective_training_algorithm,
    training_requires_action_masks,
    validate_training_algorithm_config,
)
from tests.support.fakes import SyntheticBackend


def test_resolve_effective_training_algorithm_uses_maskable_auto_mode(
    tmp_path: Path,
) -> None:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    core_path.touch()
    rom_path.touch()

    config = TrainAppConfig(
        emulator=EmulatorConfig(core_path=core_path, rom_path=rom_path),
        env=EnvConfig(),
        policy=PolicyConfig(),
        curriculum=CurriculumConfig(),
        train=TrainConfig(algorithm="auto"),
    )

    assert training_requires_action_masks(config) is True
    assert (
        resolve_effective_training_algorithm(
            train_config=config.train,
        )
        == "maskable_ppo"
    )


def test_training_requires_no_action_masks_for_sac(tmp_path: Path) -> None:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    core_path.touch()
    rom_path.touch()

    config = TrainAppConfig(
        emulator=EmulatorConfig(core_path=core_path, rom_path=rom_path),
        env=EnvConfig(action=ActionConfig(name="continuous_steer_drive_lean")),
        policy=PolicyConfig(),
        curriculum=CurriculumConfig(),
        train=TrainConfig(algorithm="sac", ent_coef="auto"),
    )

    assert training_requires_action_masks(config) is False
    assert (
        resolve_effective_training_algorithm(
            train_config=config.train,
        )
        == "sac"
    )


def test_training_requires_no_action_masks_for_hybrid_action_sac(tmp_path: Path) -> None:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    core_path.touch()
    rom_path.touch()

    config = TrainAppConfig(
        emulator=EmulatorConfig(core_path=core_path, rom_path=rom_path),
        env=EnvConfig(action=ActionConfig(name="hybrid_steer_drive_boost_lean")),
        policy=PolicyConfig(),
        curriculum=CurriculumConfig(),
        train=TrainConfig(algorithm="hybrid_action_sac", ent_coef="auto"),
    )

    assert training_requires_action_masks(config) is False
    assert (
        resolve_effective_training_algorithm(
            train_config=config.train,
        )
        == "hybrid_action_sac"
    )


def test_training_requires_action_masks_for_maskable_hybrid_action_sac(
    tmp_path: Path,
) -> None:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    core_path.touch()
    rom_path.touch()

    config = TrainAppConfig(
        emulator=EmulatorConfig(core_path=core_path, rom_path=rom_path),
        env=EnvConfig(action=ActionConfig(name="hybrid_steer_drive_boost_lean")),
        policy=PolicyConfig(),
        curriculum=CurriculumConfig(),
        train=TrainConfig(algorithm="maskable_hybrid_action_sac", ent_coef="auto"),
    )

    assert training_requires_action_masks(config) is True
    assert (
        resolve_effective_training_algorithm(
            train_config=config.train,
        )
        == "maskable_hybrid_action_sac"
    )


def test_training_requires_action_masks_for_maskable_hybrid_action_ppo(
    tmp_path: Path,
) -> None:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    core_path.touch()
    rom_path.touch()

    config = TrainAppConfig(
        emulator=EmulatorConfig(core_path=core_path, rom_path=rom_path),
        env=EnvConfig(action=ActionConfig(name="hybrid_steer_drive_boost_lean")),
        policy=PolicyConfig(),
        curriculum=CurriculumConfig(),
        train=TrainConfig(algorithm="maskable_hybrid_action_ppo"),
    )

    assert training_requires_action_masks(config) is True
    assert (
        resolve_effective_training_algorithm(
            train_config=config.train,
        )
        == "maskable_hybrid_action_ppo"
    )


def test_training_requires_action_masks_for_maskable_hybrid_recurrent_ppo(
    tmp_path: Path,
) -> None:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    core_path.touch()
    rom_path.touch()

    config = TrainAppConfig(
        emulator=EmulatorConfig(core_path=core_path, rom_path=rom_path),
        env=EnvConfig(action=ActionConfig(name="hybrid_steer_drive_boost_lean")),
        policy=PolicyConfig(recurrent=PolicyRecurrentConfig(enabled=True)),
        curriculum=CurriculumConfig(),
        train=TrainConfig(algorithm="maskable_hybrid_recurrent_ppo"),
    )

    assert training_requires_action_masks(config) is True
    assert (
        resolve_effective_training_algorithm(
            train_config=config.train,
        )
        == "maskable_hybrid_recurrent_ppo"
    )


def test_validate_training_algorithm_config_rejects_sac_without_continuous_action(
    tmp_path: Path,
) -> None:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    core_path.touch()
    rom_path.touch()

    config = TrainAppConfig(
        emulator=EmulatorConfig(core_path=core_path, rom_path=rom_path),
        env=EnvConfig(action=ActionConfig(name="steer_drive")),
        policy=PolicyConfig(),
        curriculum=CurriculumConfig(),
        train=TrainConfig(algorithm="sac", ent_coef="auto"),
    )

    with pytest.raises(RuntimeError, match="continuous steer-drive action adapter"):
        validate_training_algorithm_config(config)


def test_validate_training_algorithm_config_rejects_maskable_hybrid_ppo_without_hybrid_action(
    tmp_path: Path,
) -> None:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    core_path.touch()
    rom_path.touch()

    config = TrainAppConfig(
        emulator=EmulatorConfig(core_path=core_path, rom_path=rom_path),
        env=EnvConfig(action=ActionConfig(name="continuous_steer_drive")),
        policy=PolicyConfig(),
        curriculum=CurriculumConfig(),
        train=TrainConfig(algorithm="maskable_hybrid_action_ppo"),
    )

    with pytest.raises(RuntimeError, match="hybrid steer-drive action adapter"):
        validate_training_algorithm_config(config)


def test_validate_training_algorithm_config_rejects_hybrid_sac_masks(
    tmp_path: Path,
) -> None:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    core_path.touch()
    rom_path.touch()

    config = TrainAppConfig(
        emulator=EmulatorConfig(core_path=core_path, rom_path=rom_path),
        env=EnvConfig(
            action=ActionConfig(
                name="hybrid_steer_drive_boost_lean",
                mask=ActionMaskConfig(boost=(0,)),
            )
        ),
        policy=PolicyConfig(),
        curriculum=CurriculumConfig(),
        train=TrainConfig(algorithm="hybrid_action_sac", ent_coef="auto"),
    )

    with pytest.raises(RuntimeError, match="not maskable yet"):
        validate_training_algorithm_config(config)


def test_validate_training_algorithm_config_accepts_maskable_hybrid_sac_masks(
    tmp_path: Path,
) -> None:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    core_path.touch()
    rom_path.touch()

    config = TrainAppConfig(
        emulator=EmulatorConfig(core_path=core_path, rom_path=rom_path),
        env=EnvConfig(
            action=ActionConfig(
                name="hybrid_steer_drive_boost_lean",
                mask=ActionMaskConfig(boost=(0,), lean=(0, 1, 2)),
            )
        ),
        policy=PolicyConfig(),
        curriculum=CurriculumConfig(),
        train=TrainConfig(algorithm="maskable_hybrid_action_sac", ent_coef="auto"),
    )

    validate_training_algorithm_config(config)


def test_build_ppo_model_can_construct_maskable_ppo() -> None:
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    env = DummyVecEnv(
        [
            lambda: FZeroXEnv(
                backend=SyntheticBackend(),
                config=EnvConfig(action=ActionConfig(mask=ActionMaskConfig(lean=(0,)))),
            )
        ]
    )

    try:
        model = build_ppo_model(
            train_env=env,
            train_config=TrainConfig(algorithm="auto"),
            policy_config=PolicyConfig(),
            tensorboard_log=None,
        )
    finally:
        env.close()

    assert isinstance(model, MaskablePPO)


def test_build_training_model_can_construct_sac() -> None:
    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import DummyVecEnv

    env = DummyVecEnv(
        [
            lambda: FZeroXEnv(
                backend=SyntheticBackend(),
                config=EnvConfig(
                    action=ActionConfig(name="continuous_steer_drive_lean"),
                    observation=ObservationConfig(mode="image_state"),
                ),
            )
        ]
    )

    try:
        model = build_training_model(
            train_env=env,
            train_config=TrainConfig(
                algorithm="sac",
                buffer_size=4,
                learning_starts=0,
                ent_coef="auto",
                device="cpu",
            ),
            policy_config=PolicyConfig(),
            tensorboard_log=None,
        )
    finally:
        env.close()

    assert isinstance(model, SAC)


def test_build_training_model_can_construct_hybrid_action_sac() -> None:
    from sb3x import HybridActionSAC
    from stable_baselines3.common.vec_env import DummyVecEnv

    env = DummyVecEnv(
        [
            lambda: FZeroXEnv(
                backend=SyntheticBackend(),
                config=EnvConfig(
                    action=ActionConfig(name="hybrid_steer_drive_boost_lean"),
                    observation=ObservationConfig(mode="image_state"),
                ),
            )
        ]
    )

    try:
        model = build_training_model(
            train_env=env,
            train_config=TrainConfig(
                algorithm="hybrid_action_sac",
                buffer_size=4,
                learning_starts=0,
                ent_coef="auto",
                device="cpu",
            ),
            policy_config=PolicyConfig(),
            tensorboard_log=None,
        )
    finally:
        env.close()

    assert isinstance(model, HybridActionSAC)


def test_build_training_model_can_construct_maskable_hybrid_action_sac() -> None:
    from sb3x import MaskableHybridActionSAC
    from stable_baselines3.common.vec_env import DummyVecEnv

    env = DummyVecEnv(
        [
            lambda: FZeroXEnv(
                backend=SyntheticBackend(),
                config=EnvConfig(
                    action=ActionConfig(name="hybrid_steer_drive_boost_lean"),
                    observation=ObservationConfig(mode="image_state"),
                ),
            )
        ]
    )

    try:
        model = build_training_model(
            train_env=env,
            train_config=TrainConfig(
                algorithm="maskable_hybrid_action_sac",
                buffer_size=4,
                learning_starts=0,
                ent_coef="auto",
                device="cpu",
            ),
            policy_config=PolicyConfig(),
            tensorboard_log=None,
        )
    finally:
        env.close()

    assert isinstance(model, MaskableHybridActionSAC)


def test_build_ppo_model_can_construct_maskable_hybrid_action_ppo() -> None:
    from sb3x import MaskableHybridActionPPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    env = DummyVecEnv(
        [
            lambda: FZeroXEnv(
                backend=SyntheticBackend(),
                config=EnvConfig(
                    action=ActionConfig(name="hybrid_steer_drive_boost_lean"),
                    observation=ObservationConfig(mode="image_state"),
                ),
            )
        ]
    )

    try:
        model = build_ppo_model(
            train_env=env,
            train_config=TrainConfig(
                algorithm="maskable_hybrid_action_ppo",
                n_steps=4,
                batch_size=4,
                device="cpu",
            ),
            policy_config=PolicyConfig(),
            tensorboard_log=None,
        )
    finally:
        env.close()

    assert isinstance(model, MaskableHybridActionPPO)


def test_build_ppo_model_can_construct_maskable_hybrid_recurrent_ppo() -> None:
    from sb3x import MaskableHybridRecurrentPPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    env = DummyVecEnv(
        [
            lambda: FZeroXEnv(
                backend=SyntheticBackend(),
                config=EnvConfig(
                    action=ActionConfig(name="hybrid_steer_drive_boost_lean"),
                    observation=ObservationConfig(mode="image_state"),
                ),
            )
        ]
    )

    try:
        model = build_ppo_model(
            train_env=env,
            train_config=TrainConfig(
                algorithm="maskable_hybrid_recurrent_ppo",
                n_steps=4,
                batch_size=4,
                device="cpu",
            ),
            policy_config=PolicyConfig(
                recurrent=PolicyRecurrentConfig(
                    enabled=True,
                    hidden_size=512,
                    n_lstm_layers=1,
                )
            ),
            tensorboard_log=None,
        )
    finally:
        env.close()

    assert isinstance(model, MaskableHybridRecurrentPPO)
    assert model.policy.state_dict()["lstm_actor.weight_ih_l0"].shape[0] == 4 * 512


def test_build_ppo_model_rejects_recurrent_policy_with_feedforward_algorithm() -> None:
    from stable_baselines3.common.vec_env import DummyVecEnv

    env = DummyVecEnv(
        [
            lambda: FZeroXEnv(
                backend=SyntheticBackend(),
                config=EnvConfig(action=ActionConfig(mask=ActionMaskConfig(lean=(0,)))),
            )
        ]
    )

    try:
        with pytest.raises(
            RuntimeError,
            match="Recurrent policy config requires a recurrent train.algorithm",
        ):
            build_ppo_model(
                train_env=env,
                train_config=TrainConfig(algorithm="maskable_ppo"),
                policy_config=PolicyConfig(
                    recurrent=PolicyRecurrentConfig(enabled=True),
                ),
                tensorboard_log=None,
            )
    finally:
        env.close()


def test_build_ppo_model_can_construct_maskable_recurrent_ppo() -> None:
    from sb3x import MaskableRecurrentPPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    env = DummyVecEnv(
        [
            lambda: FZeroXEnv(
                backend=SyntheticBackend(),
                config=EnvConfig(
                    action=ActionConfig(mask=ActionMaskConfig(lean=(0,))),
                    observation=ObservationConfig(mode="image_state"),
                ),
            )
        ]
    )

    try:
        model = build_ppo_model(
            train_env=env,
            train_config=TrainConfig(algorithm="maskable_recurrent_ppo"),
            policy_config=PolicyConfig(
                recurrent=PolicyRecurrentConfig(
                    enabled=True,
                    hidden_size=512,
                    n_lstm_layers=1,
                )
            ),
            tensorboard_log=None,
        )
    finally:
        env.close()

    assert isinstance(model, MaskableRecurrentPPO)
    assert model.policy.state_dict()["lstm_actor.weight_ih_l0"].shape[0] == 4 * 512
