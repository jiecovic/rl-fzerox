# tests/core/envs/test_action_spaces.py
from gymnasium.spaces import Box, Dict, MultiDiscrete

from rl_fzerox.core.envs import FZeroXEnv
from rl_fzerox.core.envs.engine.controls import (
    action_branch_non_neutral_allowed,
    action_branch_value_allowed,
)
from rl_fzerox.core.runtime_spec.schema import ActionMaskConfig, EnvConfig
from tests.support.action_configs import (
    configured_discrete_action,
    configured_hybrid_action,
)
from tests.support.fakes import SyntheticBackend


def test_configured_discrete_gas_boost_lean_env_exposes_four_head_action_space() -> None:
    env = FZeroXEnv(
        backend=SyntheticBackend(),
        config=EnvConfig(action=configured_discrete_action("steer", "gas", "boost", "lean")),
    )

    assert isinstance(env.action_space, MultiDiscrete)
    assert env.action_space.nvec.tolist() == [7, 2, 2, 3]


def test_configured_discrete_gas_boost_env_exposes_three_head_action_space() -> None:
    env = FZeroXEnv(
        backend=SyntheticBackend(),
        config=EnvConfig(action=configured_discrete_action("steer", "gas", "boost")),
    )

    assert isinstance(env.action_space, MultiDiscrete)
    assert env.action_space.nvec.tolist() == [7, 2, 2]


def test_configured_discrete_parallel_button_env_exposes_maskable_space() -> None:
    env = FZeroXEnv(
        backend=SyntheticBackend(),
        config=EnvConfig(
            action=configured_discrete_action(
                "steer",
                "gas",
                "air_brake",
                "boost",
                "lean",
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


def test_configured_hybrid_steer_gas_air_brake_boost_lean_env_exposes_maskable_space() -> None:
    env = FZeroXEnv(
        backend=SyntheticBackend(),
        config=EnvConfig(
            action=configured_hybrid_action(
                continuous_axes=("steer",),
                discrete_axes=("gas", "air_brake", "boost", "lean"),
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
    assert env.action_mask_branches() == {
        "gas": (False, True),
        "air_brake": (True, False),
        "boost": (True, False),
        "lean": (True, False, False),
    }
    snapshot = env.action_mask_snapshot()
    assert snapshot.flat.tolist() == env.action_masks().tolist()
    assert snapshot.branches == env.action_mask_branches()


def test_action_branch_allowed_helpers_make_missing_branch_policy_explicit() -> None:
    branches = {"boost": (True, False), "pitch": (False, False, True, False, False)}

    assert not action_branch_value_allowed(branches, "boost", 1, missing_allowed=False)
    assert action_branch_value_allowed(branches, "gas", 1, missing_allowed=True)
    assert not action_branch_value_allowed(branches, "lean", 1, missing_allowed=False)
    assert not action_branch_non_neutral_allowed(
        branches,
        "pitch",
        neutral_index=2,
        missing_allowed=False,
    )
    assert action_branch_non_neutral_allowed(
        branches,
        "gas",
        neutral_index=0,
        missing_allowed=True,
    )


def test_configured_hybrid_steer_gas_boost_lean_env_exposes_maskable_space() -> None:
    env = FZeroXEnv(
        backend=SyntheticBackend(),
        config=EnvConfig(
            action=configured_hybrid_action(
                continuous_axes=("steer",),
                discrete_axes=("gas", "boost", "lean"),
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


def test_configured_hybrid_steer_drive_env_exposes_dict_action_space() -> None:
    env = FZeroXEnv(
        backend=SyntheticBackend(),
        config=EnvConfig(
            action=configured_hybrid_action(
                continuous_axes=("steer", "drive"),
                discrete_axes=(),
            )
        ),
    )

    assert isinstance(env.action_space, Dict)
    assert isinstance(env.action_space.spaces["continuous"], Box)
    assert isinstance(env.action_space.spaces["discrete"], MultiDiscrete)
    assert env.action_space.spaces["continuous"].shape == (2,)
    assert env.action_space.spaces["discrete"].nvec.tolist() == []
    assert env.action_masks().tolist() == []


def test_configured_hybrid_boost_lean_env_exposes_boost_mask_branch() -> None:
    env = FZeroXEnv(
        backend=SyntheticBackend(),
        config=EnvConfig(
            action=configured_hybrid_action(
                continuous_axes=("steer", "drive"),
                discrete_axes=("boost", "lean"),
            )
        ),
    )

    assert isinstance(env.action_space, Dict)
    assert isinstance(env.action_space.spaces["continuous"], Box)
    assert isinstance(env.action_space.spaces["discrete"], MultiDiscrete)
    assert env.action_space.spaces["continuous"].shape == (2,)
    assert env.action_space.spaces["discrete"].nvec.tolist() == [2, 3]
    assert env.action_masks().tolist() == [True, True, True, True, True]
