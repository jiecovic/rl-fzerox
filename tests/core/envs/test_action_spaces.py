# tests/core/envs/test_action_spaces.py
from gymnasium.spaces import Box, Dict, MultiDiscrete

from rl_fzerox.core.config.schema import ActionConfig, ActionMaskConfig, EnvConfig
from rl_fzerox.core.envs import FZeroXEnv
from tests.support.fakes import SyntheticBackend


def test_extended_action_env_exposes_four_head_action_space() -> None:
    env = FZeroXEnv(
        backend=SyntheticBackend(),
        config=EnvConfig(action=ActionConfig(name="steer_drive_boost_lean")),
    )

    assert isinstance(env.action_space, MultiDiscrete)
    assert env.action_space.nvec.tolist() == [7, 3, 2, 3]


def test_boost_action_env_exposes_three_head_action_space() -> None:
    env = FZeroXEnv(
        backend=SyntheticBackend(),
        config=EnvConfig(action=ActionConfig(name="steer_drive_boost")),
    )

    assert isinstance(env.action_space, MultiDiscrete)
    assert env.action_space.nvec.tolist() == [7, 3, 2]


def test_steer_gas_air_brake_boost_lean_env_exposes_maskable_full_action_space() -> None:
    env = FZeroXEnv(
        backend=SyntheticBackend(),
        config=EnvConfig(
            action=ActionConfig(
                name="steer_gas_air_brake_boost_lean",
                steer_buckets=3,
                mask=ActionMaskConfig(boost=(0,), lean=(0,)),
            )
        ),
    )

    assert isinstance(env.action_space, MultiDiscrete)
    assert env.action_space.nvec.tolist() == [3, 2, 2, 2, 3]
    assert env.action_masks().tolist() == [
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        False,
        True,
        False,
        False,
    ]


def test_hybrid_steer_gas_air_brake_boost_lean_env_exposes_maskable_action_space() -> None:
    env = FZeroXEnv(
        backend=SyntheticBackend(),
        config=EnvConfig(
            action=ActionConfig(
                name="hybrid_steer_gas_air_brake_boost_lean",
                mask=ActionMaskConfig(gas=(1,), air_brake=(0,), boost=(0,), lean=(0,)),
            )
        ),
    )

    assert isinstance(env.action_space, Dict)
    assert isinstance(env.action_space.spaces["continuous"], Box)
    assert env.action_space.spaces["continuous"].shape == (1,)
    assert isinstance(env.action_space.spaces["discrete"], MultiDiscrete)
    assert env.action_space.spaces["discrete"].nvec.tolist() == [2, 2, 2, 3]
    assert env.action_masks().tolist() == [
        False,
        True,
        True,
        False,
        True,
        False,
        True,
        False,
        False,
    ]


def test_hybrid_steer_gas_boost_lean_env_exposes_maskable_action_space() -> None:
    env = FZeroXEnv(
        backend=SyntheticBackend(),
        config=EnvConfig(
            action=ActionConfig(
                name="hybrid_steer_gas_boost_lean",
                mask=ActionMaskConfig(gas=(1,), boost=(0,), lean=(0,)),
            )
        ),
    )

    assert isinstance(env.action_space, Dict)
    assert isinstance(env.action_space.spaces["continuous"], Box)
    assert env.action_space.spaces["continuous"].shape == (1,)
    assert isinstance(env.action_space.spaces["discrete"], MultiDiscrete)
    assert env.action_space.spaces["discrete"].nvec.tolist() == [2, 2, 3]
    assert env.action_masks().tolist() == [
        False,
        True,
        True,
        False,
        True,
        False,
        False,
    ]


def test_continuous_action_env_exposes_box_action_space() -> None:
    env = FZeroXEnv(
        backend=SyntheticBackend(),
        config=EnvConfig(action=ActionConfig(name="continuous_steer_drive")),
    )

    assert isinstance(env.action_space, Box)
    assert env.action_space.shape == (2,)
    assert env.action_masks().tolist() == []


def test_continuous_lean_action_env_exposes_box_action_space() -> None:
    env = FZeroXEnv(
        backend=SyntheticBackend(),
        config=EnvConfig(action=ActionConfig(name="continuous_steer_drive_lean")),
    )

    assert isinstance(env.action_space, Box)
    assert env.action_space.shape == (3,)
    assert env.action_masks().tolist() == []


def test_hybrid_lean_action_env_exposes_dict_action_space() -> None:
    env = FZeroXEnv(
        backend=SyntheticBackend(),
        config=EnvConfig(action=ActionConfig(name="hybrid_steer_drive_lean")),
    )

    assert isinstance(env.action_space, Dict)
    assert isinstance(env.action_space.spaces["continuous"], Box)
    assert isinstance(env.action_space.spaces["discrete"], MultiDiscrete)
    assert env.action_space.spaces["continuous"].shape == (2,)
    assert env.action_space.spaces["discrete"].nvec.tolist() == [3]
    assert env.action_masks().tolist() == [True, True, True]


def test_hybrid_boost_lean_action_env_exposes_boost_mask_branch() -> None:
    env = FZeroXEnv(
        backend=SyntheticBackend(),
        config=EnvConfig(action=ActionConfig(name="hybrid_steer_drive_boost_lean")),
    )

    assert isinstance(env.action_space, Dict)
    assert isinstance(env.action_space.spaces["continuous"], Box)
    assert isinstance(env.action_space.spaces["discrete"], MultiDiscrete)
    assert env.action_space.spaces["continuous"].shape == (2,)
    assert env.action_space.spaces["discrete"].nvec.tolist() == [3, 2]
    assert env.action_masks().tolist() == [True, True, True, True, True]


def test_hybrid_boost_lean_primitive_env_masks_future_primitives_by_default() -> None:
    env = FZeroXEnv(
        backend=SyntheticBackend(),
        config=EnvConfig(action=ActionConfig(name="hybrid_steer_drive_boost_lean_primitive")),
    )

    assert isinstance(env.action_space, Dict)
    assert isinstance(env.action_space.spaces["continuous"], Box)
    assert isinstance(env.action_space.spaces["discrete"], MultiDiscrete)
    assert env.action_space.spaces["continuous"].shape == (3,)
    assert env.action_space.spaces["discrete"].nvec.tolist() == [7, 2]
    assert env.action_masks().tolist() == [
        True,
        True,
        True,
        False,
        False,
        False,
        False,
        True,
        True,
    ]
