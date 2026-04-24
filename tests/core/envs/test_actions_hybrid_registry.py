# tests/core/envs/test_actions_hybrid_registry.py
import numpy as np
import pytest
from gymnasium.spaces import Box, Dict, MultiDiscrete

from fzerox_emulator import ControllerState
from rl_fzerox.core.config.schema import ActionConfig, ActionMaskConfig
from rl_fzerox.core.envs.actions import (
    ACCELERATE_MASK,
    ACTION_ADAPTER_REGISTRY,
    AIR_BRAKE_MASK,
    BOOST_MASK,
    LEAN_LEFT_MASK,
    LEAN_RIGHT_MASK,
    ContinuousSteerDriveActionAdapter,
    HybridSteerDriveAirBrakeBoostLeanPitchActionAdapter,
    HybridSteerDriveBoostLeanActionAdapter,
    HybridSteerDriveBoostLeanPrimitiveActionAdapter,
    HybridSteerDriveLeanActionAdapter,
    HybridSteerGasAirBrakeBoostLeanActionAdapter,
    HybridSteerGasBoostLeanActionAdapter,
    SteerDriveBoostActionAdapter,
    SteerDriveBoostLeanActionAdapter,
    SteerGasAirBrakeBoostLeanActionAdapter,
    action_adapter_names,
    build_action_adapter,
)


def test_hybrid_steer_drive_lean_adapter_uses_dict_action_space() -> None:
    adapter = HybridSteerDriveLeanActionAdapter(ActionConfig(name="hybrid_steer_drive_lean"))

    assert isinstance(adapter.action_space, Dict)
    continuous_space = adapter.action_space.spaces["continuous"]
    discrete_space = adapter.action_space.spaces["discrete"]
    assert isinstance(continuous_space, Box)
    assert isinstance(discrete_space, MultiDiscrete)
    assert continuous_space.shape == (2,)
    assert continuous_space.dtype == np.float32
    assert discrete_space.nvec.tolist() == [3]
    assert np.array_equal(adapter.idle_action["continuous"], np.zeros(2, dtype=np.float32))
    assert np.array_equal(adapter.idle_action["discrete"], np.zeros(1, dtype=np.int64))
    assert adapter.action_mask().tolist() == [True, True, True]


def test_hybrid_steer_drive_lean_adapter_decodes_discrete_lean_branch() -> None:
    adapter = HybridSteerDriveLeanActionAdapter(
        ActionConfig(name="hybrid_steer_drive_lean", continuous_drive_deadzone=0.2)
    )

    control_state = adapter.decode(
        {
            "continuous": np.array([0.25, 0.7], dtype=np.float32),
            "discrete": np.array([1], dtype=np.int64),
        }
    )

    assert control_state == ControllerState(
        joypad_mask=ACCELERATE_MASK | LEAN_LEFT_MASK,
        left_stick_x=0.25,
    )


def test_hybrid_steer_drive_boost_lean_adapter_decodes_boost_branch() -> None:
    adapter = HybridSteerDriveBoostLeanActionAdapter(
        ActionConfig(name="hybrid_steer_drive_boost_lean", continuous_drive_deadzone=0.2)
    )

    control_state = adapter.decode(
        {
            "continuous": np.array([0.25, 0.7], dtype=np.float32),
            "discrete": np.array([2, 1], dtype=np.int64),
        }
    )

    discrete_space = adapter.action_space.spaces["discrete"]
    assert isinstance(discrete_space, MultiDiscrete)
    assert discrete_space.nvec.tolist() == [3, 2]
    assert np.array_equal(adapter.idle_action["discrete"], np.zeros(2, dtype=np.int64))
    assert control_state == ControllerState(
        joypad_mask=ACCELERATE_MASK | LEAN_RIGHT_MASK | BOOST_MASK,
        left_stick_x=0.25,
    )


def test_hybrid_steer_drive_boost_lean_adapter_masks_boost_branch() -> None:
    adapter = HybridSteerDriveBoostLeanActionAdapter(
        ActionConfig(name="hybrid_steer_drive_boost_lean")
    )

    mask = adapter.action_mask(dynamic_overrides={"boost": (0,)})

    assert mask.tolist() == [True, True, True, True, False]


def test_hybrid_steer_drive_boost_lean_primitive_adapter_reserves_future_primitives() -> None:
    adapter = HybridSteerDriveBoostLeanPrimitiveActionAdapter(
        ActionConfig(name="hybrid_steer_drive_boost_lean_primitive")
    )

    continuous_space = adapter.action_space.spaces["continuous"]
    discrete_space = adapter.action_space.spaces["discrete"]
    assert isinstance(continuous_space, Box)
    assert isinstance(discrete_space, MultiDiscrete)
    assert continuous_space.shape == (3,)
    assert discrete_space.nvec.tolist() == [7, 2]
    assert np.array_equal(adapter.idle_action["continuous"], np.zeros(3, dtype=np.float32))
    assert np.array_equal(adapter.idle_action["discrete"], np.zeros(2, dtype=np.int64))
    assert adapter.action_mask().tolist() == [
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


def test_hybrid_steer_drive_boost_lean_primitive_adapter_decodes_current_lean_values() -> None:
    adapter = HybridSteerDriveBoostLeanPrimitiveActionAdapter(
        ActionConfig(
            name="hybrid_steer_drive_boost_lean_primitive",
            continuous_drive_deadzone=0.2,
        )
    )

    control_state = adapter.decode(
        {
            "continuous": np.array([0.25, 0.7, -1.0], dtype=np.float32),
            "discrete": np.array([2, 1], dtype=np.int64),
        }
    )

    assert control_state == ControllerState(
        joypad_mask=ACCELERATE_MASK | LEAN_RIGHT_MASK | BOOST_MASK,
        left_stick_x=0.25,
    )


def test_hybrid_steer_drive_boost_lean_primitive_adapter_keeps_reserved_values_noop() -> None:
    adapter = HybridSteerDriveBoostLeanPrimitiveActionAdapter(
        ActionConfig(
            name="hybrid_steer_drive_boost_lean_primitive",
            continuous_drive_deadzone=0.2,
        )
    )

    control_state = adapter.decode(
        {
            "continuous": np.array([0.25, 0.7, -1.0], dtype=np.float32),
            "discrete": np.array([3, 0], dtype=np.int64),
        }
    )

    assert control_state == ControllerState(
        joypad_mask=ACCELERATE_MASK,
        left_stick_x=0.25,
    )


def test_hybrid_steer_drive_boost_lean_primitive_adapter_decodes_air_brake_axis() -> None:
    adapter = HybridSteerDriveBoostLeanPrimitiveActionAdapter(
        ActionConfig(
            name="hybrid_steer_drive_boost_lean_primitive",
            continuous_drive_mode="pwm",
            continuous_drive_deadzone=0.0,
        )
    )

    control_state = adapter.decode(
        {
            "continuous": np.array([0.25, -1.0, 1.0], dtype=np.float32),
            "discrete": np.array([0, 0], dtype=np.int64),
        }
    )

    assert control_state == ControllerState(
        joypad_mask=AIR_BRAKE_MASK,
        left_stick_x=0.25,
    )


def test_hybrid_steer_drive_boost_lean_primitive_adapter_can_disable_air_brake() -> None:
    adapter = HybridSteerDriveBoostLeanPrimitiveActionAdapter(
        ActionConfig(
            name="hybrid_steer_drive_boost_lean_primitive",
            continuous_drive_mode="pwm",
            continuous_drive_deadzone=0.0,
            continuous_air_brake_mode="off",
        )
    )

    control_state = adapter.decode(
        {
            "continuous": np.array([0.25, -1.0, 1.0], dtype=np.float32),
            "discrete": np.array([0, 0], dtype=np.int64),
        }
    )

    assert control_state == ControllerState(joypad_mask=0, left_stick_x=0.25)


def test_hybrid_steer_drive_boost_lean_primitive_adapter_keeps_neutral_air_brake_off() -> None:
    adapter = HybridSteerDriveBoostLeanPrimitiveActionAdapter(
        ActionConfig(
            name="hybrid_steer_drive_boost_lean_primitive",
            continuous_drive_mode="pwm",
            continuous_drive_deadzone=0.0,
        )
    )

    control_state = adapter.decode(
        {
            "continuous": np.array([0.25, -1.0, 0.0], dtype=np.float32),
            "discrete": np.array([0, 0], dtype=np.int64),
        }
    )

    assert control_state == ControllerState(joypad_mask=0, left_stick_x=0.25)


def test_hybrid_steer_drive_boost_lean_primitive_adapter_allows_config_unmask() -> None:
    adapter = HybridSteerDriveBoostLeanPrimitiveActionAdapter(
        ActionConfig(name="hybrid_steer_drive_boost_lean_primitive")
    )

    mask = adapter.action_mask(base_overrides={"lean": (0, 3), "boost": (0,)})

    assert mask.tolist() == [
        True,
        False,
        False,
        True,
        False,
        False,
        False,
        True,
        False,
    ]


def test_build_action_adapter_uses_basic_adapter_by_default() -> None:
    adapter = build_action_adapter(ActionConfig())

    assert isinstance(adapter, SteerDriveBoostLeanActionAdapter)


def test_extended_adapter_uses_four_head_multidiscrete_space() -> None:
    adapter = SteerDriveBoostLeanActionAdapter(
        ActionConfig(name="steer_drive_boost_lean", steer_buckets=7)
    )

    assert isinstance(adapter.action_space, MultiDiscrete)
    assert adapter.action_space.nvec.tolist() == [7, 3, 2, 3]
    assert np.array_equal(adapter.idle_action, np.array([3, 0, 0, 0], dtype=np.int64))


def test_boost_adapter_uses_three_head_multidiscrete_space() -> None:
    adapter = SteerDriveBoostActionAdapter(ActionConfig(name="steer_drive_boost", steer_buckets=7))

    assert isinstance(adapter.action_space, MultiDiscrete)
    assert adapter.action_space.nvec.tolist() == [7, 3, 2]
    assert np.array_equal(adapter.idle_action, np.array([3, 0, 0], dtype=np.int64))


def test_boost_adapter_decodes_accelerate_and_boost() -> None:
    adapter = SteerDriveBoostActionAdapter(ActionConfig(name="steer_drive_boost", steer_buckets=7))

    control_state = adapter.decode(np.array([4, 1, 1], dtype=np.int64))

    assert control_state.joypad_mask == (ACCELERATE_MASK | BOOST_MASK)
    assert control_state.left_stick_x == pytest.approx(1.0 / 3.0)


def test_boost_adapter_decodes_air_brake_without_boost() -> None:
    adapter = SteerDriveBoostActionAdapter(ActionConfig(name="steer_drive_boost", steer_buckets=7))

    control_state = adapter.decode(np.array([3, 2, 0], dtype=np.int64))

    assert control_state.joypad_mask == AIR_BRAKE_MASK
    assert control_state.left_stick_x == 0.0


def test_boost_adapter_rejects_wrong_action_shape() -> None:
    adapter = SteerDriveBoostActionAdapter(ActionConfig(name="steer_drive_boost", steer_buckets=7))

    with pytest.raises(ValueError):
        adapter.decode(np.array([3, 1], dtype=np.int64))


def test_extended_adapter_decodes_accelerate_boost_and_explicit_right_lean() -> None:
    adapter = SteerDriveBoostLeanActionAdapter(
        ActionConfig(name="steer_drive_boost_lean", steer_buckets=7)
    )

    control_state = adapter.decode(np.array([1, 1, 1, 2], dtype=np.int64))

    assert control_state.joypad_mask == (ACCELERATE_MASK | BOOST_MASK | LEAN_RIGHT_MASK)
    assert control_state.left_stick_x == pytest.approx(-2.0 / 3.0)


def test_extended_adapter_decodes_explicit_left_lean_independent_of_steer() -> None:
    adapter = SteerDriveBoostLeanActionAdapter(
        ActionConfig(name="steer_drive_boost_lean", steer_buckets=7)
    )

    control_state = adapter.decode(np.array([5, 0, 0, 1], dtype=np.int64))

    assert control_state.joypad_mask == LEAN_LEFT_MASK
    assert control_state.left_stick_x == pytest.approx(2.0 / 3.0)


def test_extended_adapter_decodes_explicit_right_lean_while_steering_straight() -> None:
    adapter = SteerDriveBoostLeanActionAdapter(
        ActionConfig(name="steer_drive_boost_lean", steer_buckets=7)
    )

    control_state = adapter.decode(np.array([3, 2, 0, 2], dtype=np.int64))

    assert control_state.joypad_mask == (AIR_BRAKE_MASK | LEAN_RIGHT_MASK)
    assert control_state.left_stick_x == 0.0


def test_extended_adapter_rejects_wrong_action_shape() -> None:
    adapter = SteerDriveBoostLeanActionAdapter(
        ActionConfig(name="steer_drive_boost_lean", steer_buckets=7)
    )

    with pytest.raises(ValueError):
        adapter.decode(np.array([3, 1, 1], dtype=np.int64))


def test_extended_adapter_rejects_removed_both_leans_index() -> None:
    adapter = SteerDriveBoostLeanActionAdapter(
        ActionConfig(name="steer_drive_boost_lean", steer_buckets=7)
    )

    with pytest.raises(ValueError, match="Invalid lean index 3"):
        adapter.decode(np.array([3, 1, 1, 3], dtype=np.int64))


def test_build_action_adapter_supports_boost_only_variant() -> None:
    adapter = build_action_adapter(ActionConfig(name="steer_drive_boost"))

    assert isinstance(adapter, SteerDriveBoostActionAdapter)


def test_build_action_adapter_supports_current_lean_variants() -> None:
    assert isinstance(
        build_action_adapter(ActionConfig(name="steer_drive_boost_lean")),
        SteerDriveBoostLeanActionAdapter,
    )
    assert isinstance(
        build_action_adapter(ActionConfig(name="steer_gas_air_brake_boost_lean")),
        SteerGasAirBrakeBoostLeanActionAdapter,
    )
    assert isinstance(
        build_action_adapter(ActionConfig(name="hybrid_steer_drive_lean")),
        HybridSteerDriveLeanActionAdapter,
    )
    assert isinstance(
        build_action_adapter(ActionConfig(name="hybrid_steer_drive_boost_lean")),
        HybridSteerDriveBoostLeanActionAdapter,
    )
    assert isinstance(
        build_action_adapter(ActionConfig(name="hybrid_steer_gas_boost_lean")),
        HybridSteerGasBoostLeanActionAdapter,
    )
    assert isinstance(
        build_action_adapter(ActionConfig(name="hybrid_steer_gas_air_brake_boost_lean")),
        HybridSteerGasAirBrakeBoostLeanActionAdapter,
    )
    assert isinstance(
        build_action_adapter(ActionConfig(name="hybrid_steer_drive_air_brake_boost_lean_pitch")),
        HybridSteerDriveAirBrakeBoostLeanPitchActionAdapter,
    )


def test_build_action_adapter_supports_steer_gas_air_brake_boost_lean_variant() -> None:
    adapter = build_action_adapter(ActionConfig(name="steer_gas_air_brake_boost_lean"))

    assert isinstance(adapter, SteerGasAirBrakeBoostLeanActionAdapter)


def test_build_action_adapter_supports_continuous_steer_drive_variant() -> None:
    adapter = build_action_adapter(ActionConfig(name="continuous_steer_drive"))

    assert isinstance(adapter, ContinuousSteerDriveActionAdapter)


def test_build_action_adapter_supports_hybrid_steer_drive_lean_variant() -> None:
    adapter = build_action_adapter(ActionConfig(name="hybrid_steer_drive_lean"))

    assert isinstance(adapter, HybridSteerDriveLeanActionAdapter)


def test_build_action_adapter_supports_hybrid_boost_lean_variant() -> None:
    adapter = build_action_adapter(ActionConfig(name="hybrid_steer_drive_boost_lean"))

    assert isinstance(adapter, HybridSteerDriveBoostLeanActionAdapter)


def test_build_action_adapter_supports_hybrid_boost_lean_primitive_variant() -> None:
    adapter = build_action_adapter(ActionConfig(name="hybrid_steer_drive_boost_lean_primitive"))

    assert isinstance(adapter, HybridSteerDriveBoostLeanPrimitiveActionAdapter)


def test_action_adapter_registry_exposes_registered_names() -> None:
    assert action_adapter_names() == tuple(ACTION_ADAPTER_REGISTRY)


def test_extended_adapter_action_mask_defaults_to_all_actions_enabled() -> None:
    adapter = SteerDriveBoostLeanActionAdapter(
        ActionConfig(name="steer_drive_boost_lean", steer_buckets=7)
    )

    mask = adapter.action_mask()

    assert mask.dtype == np.bool_
    assert mask.tolist() == ([True] * (7 + 3 + 2 + 3))


def test_extended_adapter_action_mask_can_disable_lean_branch() -> None:
    base_mask = ActionMaskConfig(lean=(0,))
    adapter = SteerDriveBoostLeanActionAdapter(
        ActionConfig(
            name="steer_drive_boost_lean",
            steer_buckets=7,
            mask=base_mask,
        )
    )

    mask = adapter.action_mask(base_overrides=base_mask.branch_overrides())

    assert mask.tolist() == (([True] * 7) + ([True] * 3) + ([True] * 2) + [True, False, False])


def test_stage_action_mask_overrides_base_mask_for_same_branch() -> None:
    base_mask = ActionMaskConfig(lean=(0,))
    adapter = SteerDriveBoostLeanActionAdapter(
        ActionConfig(
            name="steer_drive_boost_lean",
            steer_buckets=7,
            mask=base_mask,
        )
    )

    mask = adapter.action_mask(
        base_overrides=base_mask.branch_overrides(),
        stage_overrides={"lean": (0, 1, 2)},
    )

    assert mask.tolist() == ([True] * (7 + 3 + 2 + 3))
