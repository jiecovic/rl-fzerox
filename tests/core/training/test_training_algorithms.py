# tests/core/training/test_training_algorithms.py
from __future__ import annotations

from pathlib import Path

import pytest

from rl_fzerox.core.envs import FZeroXEnv
from rl_fzerox.core.runtime_spec.schema import (
    ActionMaskConfig,
    CurriculumConfig,
    EmulatorConfig,
    EnvConfig,
    ObservationConfig,
    PolicyActionBiasConfig,
    PolicyConfig,
    PolicyRecurrentConfig,
    TrainAppConfig,
    TrainConfig,
)
from rl_fzerox.core.training.session.model import (
    build_ppo_model,
    resolve_effective_training_algorithm,
    training_requires_action_masks,
    validate_training_algorithm_config,
)
from tests.support.action_configs import (
    configured_discrete_action,
    configured_hybrid_action,
)
from tests.support.fakes import SyntheticBackend, vec_env_fns


def _emulator_config(tmp_path: Path) -> EmulatorConfig:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    core_path.touch()
    rom_path.touch()
    return EmulatorConfig(core_path=core_path, rom_path=rom_path)


def test_training_requires_action_masks_for_current_supported_algorithms(
    tmp_path: Path,
) -> None:
    emulator = _emulator_config(tmp_path)
    hybrid_env = EnvConfig(
        action=configured_hybrid_action(
            continuous_axes=("steer", "drive"),
            discrete_axes=("boost", "lean"),
        )
    )
    discrete_env = EnvConfig(action=configured_discrete_action("steer", "gas", "boost", "lean"))

    assert (
        training_requires_action_masks(
            TrainAppConfig(
                emulator=emulator,
                env=hybrid_env,
                policy=PolicyConfig(),
                curriculum=CurriculumConfig(),
                train=TrainConfig(algorithm="maskable_hybrid_action_ppo"),
            )
        )
        is True
    )
    assert (
        training_requires_action_masks(
            TrainAppConfig(
                emulator=emulator,
                env=hybrid_env,
                policy=PolicyConfig(),
                curriculum=CurriculumConfig(),
                train=TrainConfig(algorithm="maskable_hybrid_action_ppo"),
            )
        )
        is True
    )
    assert (
        training_requires_action_masks(
            TrainAppConfig(
                emulator=emulator,
                env=hybrid_env,
                policy=PolicyConfig(recurrent=PolicyRecurrentConfig(enabled=True)),
                curriculum=CurriculumConfig(),
                train=TrainConfig(algorithm="maskable_hybrid_recurrent_ppo"),
            )
        )
        is True
    )
    assert (
        training_requires_action_masks(
            TrainAppConfig(
                emulator=emulator,
                env=discrete_env,
                policy=PolicyConfig(),
                curriculum=CurriculumConfig(),
                train=TrainConfig(algorithm="maskable_ppo"),
            )
        )
        is True
    )


def test_resolve_effective_training_algorithm_returns_configured_algorithm(
    tmp_path: Path,
) -> None:
    config = TrainAppConfig(
        emulator=_emulator_config(tmp_path),
        env=EnvConfig(
            action=configured_hybrid_action(
                continuous_axes=("steer", "drive"),
                discrete_axes=("boost", "lean"),
            )
        ),
        policy=PolicyConfig(),
        curriculum=CurriculumConfig(),
        train=TrainConfig(algorithm="maskable_hybrid_action_ppo"),
    )

    assert resolve_effective_training_algorithm(train_config=config.train) == (
        "maskable_hybrid_action_ppo"
    )
def test_validate_training_algorithm_config_rejects_hybrid_ppo_without_hybrid_action(
    tmp_path: Path,
) -> None:
    config = TrainAppConfig(
        emulator=_emulator_config(tmp_path),
        env=EnvConfig(action=configured_discrete_action("steer", "gas", "boost", "lean")),
        policy=PolicyConfig(),
        curriculum=CurriculumConfig(),
        train=TrainConfig(algorithm="maskable_hybrid_action_ppo"),
    )

    with pytest.raises(RuntimeError, match="configured_hybrid"):
        validate_training_algorithm_config(config)
def test_build_ppo_model_can_construct_maskable_hybrid_action_ppo() -> None:
    from sb3x import MaskableHybridActionPPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    env = DummyVecEnv(
        vec_env_fns(
            lambda: FZeroXEnv(
                backend=SyntheticBackend(),
                config=EnvConfig(
                    action=configured_hybrid_action(
                        continuous_axes=("steer", "drive"),
                        discrete_axes=("boost", "lean"),
                    ),
                    observation=ObservationConfig(
                        mode="image_state",
                        state_components=("vehicle_state",),
                    ),
                ),
            )
        )
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
        vec_env_fns(
            lambda: FZeroXEnv(
                backend=SyntheticBackend(),
                config=EnvConfig(
                    action=configured_hybrid_action(
                        continuous_axes=("steer", "drive"),
                        discrete_axes=("boost", "lean"),
                    ),
                    observation=ObservationConfig(
                        mode="image_state",
                        state_components=("vehicle_state",),
                    ),
                ),
            )
        )
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


def test_build_ppo_model_applies_hybrid_gas_on_logit_bias() -> None:
    from stable_baselines3.common.vec_env import DummyVecEnv

    env = DummyVecEnv(
        vec_env_fns(
            lambda: FZeroXEnv(
                backend=SyntheticBackend(),
                config=EnvConfig(
                    action=configured_hybrid_action(
                        continuous_axes=("steer",),
                        discrete_axes=("gas", "air_brake", "boost", "lean", "pitch"),
                    ),
                    observation=ObservationConfig(
                        mode="image_state",
                        state_components=("vehicle_state",),
                    ),
                ),
            )
        )
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
                ),
                action_bias=PolicyActionBiasConfig(gas_on_logit=0.5),
            ),
            tensorboard_log=None,
        )
    finally:
        env.close()

    bias = model.policy.state_dict()["action_net.discrete_net.bias"].detach().cpu()
    assert float(bias[1] - bias[0]) == pytest.approx(0.5)


def test_build_ppo_model_can_construct_maskable_recurrent_ppo() -> None:
    from sb3x import MaskableRecurrentPPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    env = DummyVecEnv(
        vec_env_fns(
            lambda: FZeroXEnv(
                backend=SyntheticBackend(),
                config=EnvConfig(
                    action=configured_discrete_action(
                        "steer",
                        "gas",
                        "boost",
                        "lean",
                        mask=ActionMaskConfig(lean=(0,)),
                    ),
                    observation=ObservationConfig(
                        mode="image_state",
                        state_components=("vehicle_state",),
                    ),
                ),
            )
        )
    )

    try:
        model = build_ppo_model(
            train_env=env,
            train_config=TrainConfig(
                algorithm="maskable_recurrent_ppo",
                n_steps=4,
                batch_size=4,
                device="cpu",
            ),
            policy_config=PolicyConfig(recurrent=PolicyRecurrentConfig(enabled=True)),
            tensorboard_log=None,
        )
    finally:
        env.close()

    assert isinstance(model, MaskableRecurrentPPO)
