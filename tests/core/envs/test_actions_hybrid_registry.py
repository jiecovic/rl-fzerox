# tests/core/envs/test_actions_hybrid_registry.py
import numpy as np

from rl_fzerox.core.envs.actions import (
    ACTION_ADAPTER_REGISTRY,
    ConfiguredDiscreteActionAdapter,
    ConfiguredHybridActionAdapter,
    action_adapter_names,
    build_action_adapter,
)
from tests.support.action_configs import (
    configured_discrete_action,
    configured_hybrid_action,
)


def test_action_adapter_registry_only_contains_configured_layouts() -> None:
    assert action_adapter_names() == ("configured_discrete", "configured_hybrid")
    assert action_adapter_names() == tuple(ACTION_ADAPTER_REGISTRY)


def test_build_action_adapter_supports_configured_discrete_layout() -> None:
    adapter = build_action_adapter(configured_discrete_action("steer", "gas", "boost"))

    assert isinstance(adapter, ConfiguredDiscreteActionAdapter)
    assert adapter.action_space.nvec.tolist() == [7, 2, 2]


def test_build_action_adapter_supports_configured_hybrid_layout() -> None:
    adapter = build_action_adapter(
        configured_hybrid_action(
            continuous_axes=("steer", "drive"),
            discrete_axes=("boost", "lean"),
        )
    )

    assert isinstance(adapter, ConfiguredHybridActionAdapter)
    assert adapter.action_space.spaces["continuous"].shape == (2,)
    assert adapter.action_space.spaces["discrete"].nvec.tolist() == [2, 3]


def test_configured_hybrid_four_way_lean_branch_exposes_four_values() -> None:
    adapter = build_action_adapter(
        configured_hybrid_action(
            continuous_axes=("steer",),
            discrete_axes=("lean",),
            lean_output_mode="four_way_categorical",
        )
    )

    assert isinstance(adapter, ConfiguredHybridActionAdapter)
    assert adapter.action_space.spaces["discrete"].nvec.tolist() == [4]
    assert np.array_equal(adapter.idle_action["discrete"], np.array([0], dtype=np.int64))


def test_configured_hybrid_independent_lean_exposes_two_binary_branches() -> None:
    adapter = build_action_adapter(
        configured_hybrid_action(
            continuous_axes=("steer",),
            discrete_axes=("lean_left", "lean_right"),
            lean_output_mode="independent_buttons",
        )
    )

    assert isinstance(adapter, ConfiguredHybridActionAdapter)
    assert adapter.action_space.spaces["discrete"].nvec.tolist() == [2, 2]
    assert np.array_equal(
        adapter.idle_action["discrete"],
        np.array([0, 0], dtype=np.int64),
    )
