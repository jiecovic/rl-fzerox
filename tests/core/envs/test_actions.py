# tests/core/envs/test_actions.py
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
    DRIFT_LEFT_MASK,
    DRIFT_RIGHT_MASK,
    ContinuousSteerDriveActionAdapter,
    ContinuousSteerDriveDriftActionAdapter,
    ContinuousSteerDriveShoulderActionAdapter,
    HybridSteerDriveBoostDriftActionAdapter,
    HybridSteerDriveBoostShoulderActionAdapter,
    HybridSteerDriveBoostShoulderPrimitiveActionAdapter,
    HybridSteerDriveDriftActionAdapter,
    HybridSteerDriveShoulderActionAdapter,
    HybridSteerGasAirBrakeBoostShoulderActionAdapter,
    SteerDriveActionAdapter,
    SteerDriveBoostActionAdapter,
    SteerDriveBoostDriftActionAdapter,
    SteerDriveBoostShoulderActionAdapter,
    SteerGasAirBrakeBoostDriftActionAdapter,
    SteerGasAirBrakeBoostShoulderActionAdapter,
    action_adapter_names,
    build_action_adapter,
)


def test_steer_drive_adapter_uses_default_seven_bucket_multidiscrete_space() -> None:
    adapter = SteerDriveActionAdapter(ActionConfig(name="steer_drive"))

    assert isinstance(adapter.action_space, MultiDiscrete)
    assert adapter.action_space.nvec.tolist() == [7, 3]
    assert np.array_equal(adapter.idle_action, np.array([3, 0], dtype=np.int64))


def test_steer_drive_adapter_supports_custom_bucket_counts() -> None:
    adapter = SteerDriveActionAdapter(ActionConfig(name="steer_drive", steer_buckets=7))

    assert adapter.action_space.nvec.tolist() == [7, 3]
    assert np.array_equal(adapter.idle_action, np.array([3, 0], dtype=np.int64))


def test_action_config_rejects_even_steer_bucket_counts() -> None:
    with pytest.raises(ValueError, match="steer_buckets must be odd"):
        ActionConfig(steer_buckets=6)


def test_action_config_rejects_non_positive_steer_response_power() -> None:
    with pytest.raises(ValueError):
        ActionConfig(steer_response_power=0.0)


def test_steer_drive_adapter_decodes_center_accelerate_action() -> None:
    adapter = SteerDriveActionAdapter(ActionConfig(name="steer_drive", steer_buckets=7))

    control_state = adapter.decode(np.array([3, 1], dtype=np.int64))

    assert control_state == ControllerState(
        joypad_mask=ACCELERATE_MASK,
        left_stick_x=0.0,
    )


def test_steer_drive_adapter_applies_steer_response_power_to_buckets() -> None:
    adapter = SteerDriveActionAdapter(
        ActionConfig(name="steer_drive", steer_buckets=5, steer_response_power=0.5)
    )

    control_state = adapter.decode(np.array([1, 0], dtype=np.int64))

    assert control_state.left_stick_x == pytest.approx(-(0.5**0.5))


def test_steer_drive_adapter_decodes_center_air_brake_action() -> None:
    adapter = SteerDriveActionAdapter(ActionConfig(name="steer_drive", steer_buckets=7))

    control_state = adapter.decode(np.array([3, 2], dtype=np.int64))

    assert control_state == ControllerState(
        joypad_mask=AIR_BRAKE_MASK,
        left_stick_x=0.0,
    )


def test_steer_drive_adapter_rejects_wrong_action_shape() -> None:
    adapter = SteerDriveActionAdapter(ActionConfig(name="steer_drive", steer_buckets=7))

    with pytest.raises(ValueError):
        adapter.decode(np.array([3], dtype=np.int64))


def test_steer_gas_air_brake_boost_drift_adapter_uses_full_discrete_heads() -> None:
    adapter = SteerGasAirBrakeBoostDriftActionAdapter(
        ActionConfig(name="steer_gas_air_brake_boost_drift", steer_buckets=3)
    )

    assert isinstance(adapter.action_space, MultiDiscrete)
    assert adapter.action_space.nvec.tolist() == [3, 2, 2, 2, 3]
    assert np.array_equal(adapter.idle_action, np.array([1, 0, 0, 0, 0], dtype=np.int64))
    assert adapter.action_mask().tolist() == [True] * (3 + 2 + 2 + 2 + 3)


def test_steer_gas_air_brake_boost_drift_adapter_decodes_parallel_buttons() -> None:
    adapter = SteerGasAirBrakeBoostDriftActionAdapter(
        ActionConfig(name="steer_gas_air_brake_boost_drift", steer_buckets=3)
    )

    control_state = adapter.decode(np.array([0, 1, 1, 1, 2], dtype=np.int64))

    assert control_state == ControllerState(
        joypad_mask=ACCELERATE_MASK | AIR_BRAKE_MASK | BOOST_MASK | DRIFT_RIGHT_MASK,
        left_stick_x=-1.0,
    )


def test_steer_gas_air_brake_boost_drift_adapter_masks_branches_separately() -> None:
    adapter = SteerGasAirBrakeBoostDriftActionAdapter(
        ActionConfig(name="steer_gas_air_brake_boost_drift", steer_buckets=3)
    )

    mask = adapter.action_mask(
        base_overrides={
            "gas": (1,),
            "air_brake": (0,),
            "boost": (0,),
            "shoulder": (0,),
        },
    )

    assert mask.tolist() == [
        True,
        True,
        True,
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


def test_hybrid_steer_gas_air_brake_boost_shoulder_adapter_uses_one_continuous_axis() -> None:
    adapter = HybridSteerGasAirBrakeBoostShoulderActionAdapter(
        ActionConfig(name="hybrid_steer_gas_air_brake_boost_shoulder")
    )

    assert isinstance(adapter.action_space, Dict)
    assert isinstance(adapter.action_space.spaces["continuous"], Box)
    assert adapter.action_space.spaces["continuous"].shape == (1,)
    assert isinstance(adapter.action_space.spaces["discrete"], MultiDiscrete)
    assert adapter.action_space.spaces["discrete"].nvec.tolist() == [2, 2, 2, 3]
    assert adapter.idle_action["continuous"].tolist() == [0.0]
    assert adapter.idle_action["discrete"].tolist() == [0, 0, 0, 0]
    assert tuple(dimension.label for dimension in adapter.action_dimensions) == (
        "gas",
        "air_brake",
        "boost",
        "shoulder",
    )


def test_hybrid_steer_gas_air_brake_boost_shoulder_adapter_decodes_discrete_buttons() -> None:
    adapter = HybridSteerGasAirBrakeBoostShoulderActionAdapter(
        ActionConfig(name="hybrid_steer_gas_air_brake_boost_shoulder")
    )

    control_state = adapter.decode(
        {
            "continuous": np.array([-0.75], dtype=np.float32),
            "discrete": np.array([1, 1, 1, 2], dtype=np.int64),
        }
    )

    assert control_state == ControllerState(
        joypad_mask=ACCELERATE_MASK | AIR_BRAKE_MASK | BOOST_MASK | DRIFT_RIGHT_MASK,
        left_stick_x=-0.75,
    )


def test_hybrid_steer_gas_air_brake_boost_shoulder_adapter_applies_steer_curve() -> None:
    adapter = HybridSteerGasAirBrakeBoostShoulderActionAdapter(
        ActionConfig(
            name="hybrid_steer_gas_air_brake_boost_shoulder",
            steer_response_power=0.5,
        )
    )

    control_state = adapter.decode(
        {
            "continuous": np.array([-0.25], dtype=np.float32),
            "discrete": np.array([0, 0, 0, 0], dtype=np.int64),
        }
    )

    assert control_state == ControllerState(joypad_mask=0, left_stick_x=-0.5)


def test_hybrid_steer_gas_air_brake_boost_shoulder_adapter_masks_discrete_heads() -> None:
    adapter = HybridSteerGasAirBrakeBoostShoulderActionAdapter(
        ActionConfig(name="hybrid_steer_gas_air_brake_boost_shoulder")
    )

    mask = adapter.action_mask(
        base_overrides={
            "gas": (1,),
            "air_brake": (0,),
            "boost": (0,),
            "shoulder": (0,),
        },
    )

    assert mask.tolist() == [
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


def test_continuous_steer_drive_adapter_uses_two_axis_box_space() -> None:
    adapter = ContinuousSteerDriveActionAdapter(ActionConfig(name="continuous_steer_drive"))

    assert isinstance(adapter.action_space, Box)
    assert adapter.action_space.shape == (2,)
    assert adapter.action_space.dtype == np.float32
    assert np.array_equal(adapter.action_space.low, np.array([-1.0, -1.0], dtype=np.float32))
    assert np.array_equal(adapter.action_space.high, np.array([1.0, 1.0], dtype=np.float32))
    assert np.array_equal(adapter.idle_action, np.array([0.0, 0.0], dtype=np.float32))


def test_continuous_steer_drive_adapter_decodes_accelerate_and_coast() -> None:
    adapter = ContinuousSteerDriveActionAdapter(
        ActionConfig(name="continuous_steer_drive", continuous_drive_deadzone=0.2)
    )

    accelerate_state = adapter.decode(np.array([0.5, 0.7], dtype=np.float32))
    coast_state = adapter.decode(np.array([-0.5, -0.7], dtype=np.float32))

    assert accelerate_state == ControllerState(
        joypad_mask=ACCELERATE_MASK,
        left_stick_x=0.5,
    )
    assert coast_state == ControllerState(joypad_mask=0, left_stick_x=-0.5)


def test_continuous_steer_drive_adapter_applies_steer_curve() -> None:
    adapter = ContinuousSteerDriveActionAdapter(
        ActionConfig(
            name="continuous_steer_drive",
            continuous_drive_deadzone=0.2,
            steer_response_power=0.5,
        )
    )

    control_state = adapter.decode(np.array([0.25, -1.0], dtype=np.float32))

    assert control_state == ControllerState(joypad_mask=0, left_stick_x=0.5)


def test_continuous_steer_drive_adapter_coasts_inside_drive_deadzone() -> None:
    adapter = ContinuousSteerDriveActionAdapter(
        ActionConfig(name="continuous_steer_drive", continuous_drive_deadzone=0.2)
    )

    control_state = adapter.decode(np.array([2.0, 0.1], dtype=np.float32))

    assert control_state == ControllerState(joypad_mask=0, left_stick_x=1.0)
    assert adapter.action_mask().tolist() == []


def test_continuous_steer_drive_adapter_pwm_defaults_to_full_throttle() -> None:
    adapter = ContinuousSteerDriveActionAdapter(
        ActionConfig(
            name="continuous_steer_drive",
            continuous_drive_mode="pwm",
            continuous_drive_deadzone=0.0,
        )
    )

    coast_mask = adapter.decode(np.array([0.0, -1.0], dtype=np.float32)).joypad_mask
    partial_accelerate_masks = [
        adapter.decode(np.array([0.0, -0.5], dtype=np.float32)).joypad_mask for _ in range(4)
    ]
    adapter.reset()
    neutral_accelerate_masks = [
        adapter.decode(np.array([0.0, 0.0], dtype=np.float32)).joypad_mask for _ in range(3)
    ]
    adapter.reset()
    full_accelerate_masks = [
        adapter.decode(np.array([0.0, 1.0], dtype=np.float32)).joypad_mask for _ in range(3)
    ]

    assert coast_mask == 0
    assert partial_accelerate_masks == [0, ACCELERATE_MASK, 0, ACCELERATE_MASK]
    assert neutral_accelerate_masks == [
        ACCELERATE_MASK,
        ACCELERATE_MASK,
        ACCELERATE_MASK,
    ]
    assert full_accelerate_masks == [ACCELERATE_MASK, ACCELERATE_MASK, ACCELERATE_MASK]


def test_continuous_steer_drive_adapter_can_force_full_accelerate() -> None:
    adapter = ContinuousSteerDriveActionAdapter(
        ActionConfig(
            name="continuous_steer_drive",
            continuous_drive_mode="always_accelerate",
        )
    )

    control_state = adapter.decode(np.array([-0.5, -1.0], dtype=np.float32))

    assert control_state == ControllerState(
        joypad_mask=ACCELERATE_MASK,
        left_stick_x=-0.5,
    )


def test_continuous_steer_drive_drift_adapter_uses_three_axis_box_space() -> None:
    adapter = ContinuousSteerDriveDriftActionAdapter(
        ActionConfig(name="continuous_steer_drive_drift")
    )

    assert isinstance(adapter.action_space, Box)
    assert adapter.action_space.shape == (3,)
    assert adapter.action_space.dtype == np.float32
    assert np.array_equal(
        adapter.action_space.low,
        np.array([-1.0, -1.0, -1.0], dtype=np.float32),
    )
    assert np.array_equal(
        adapter.action_space.high,
        np.array([1.0, 1.0, 1.0], dtype=np.float32),
    )
    assert np.array_equal(adapter.idle_action, np.zeros(3, dtype=np.float32))


def test_continuous_steer_drive_drift_adapter_decodes_right_drift() -> None:
    adapter = ContinuousSteerDriveDriftActionAdapter(
        ActionConfig(
            name="continuous_steer_drive_drift",
            continuous_drive_deadzone=1.0 / 3.0,
            continuous_shoulder_deadzone=1.0 / 3.0,
        )
    )

    control_state = adapter.decode(np.array([0.25, 0.75, 0.75], dtype=np.float32))

    assert control_state == ControllerState(
        joypad_mask=ACCELERATE_MASK | DRIFT_RIGHT_MASK,
        left_stick_x=0.25,
    )


def test_continuous_steer_drive_drift_adapter_negative_drive_coasts() -> None:
    adapter = ContinuousSteerDriveDriftActionAdapter(
        ActionConfig(
            name="continuous_steer_drive_drift",
            continuous_drive_deadzone=1.0 / 3.0,
            continuous_shoulder_deadzone=1.0 / 3.0,
        )
    )

    control_state = adapter.decode(np.array([-0.25, -0.75, -0.75], dtype=np.float32))

    assert control_state == ControllerState(
        joypad_mask=DRIFT_LEFT_MASK,
        left_stick_x=-0.25,
    )
    assert adapter.action_mask().tolist() == []


def test_continuous_steer_drive_drift_adapter_pwm_preserves_drift_hold() -> None:
    adapter = ContinuousSteerDriveDriftActionAdapter(
        ActionConfig(
            name="continuous_steer_drive_drift",
            continuous_drive_mode="pwm",
            continuous_drive_deadzone=0.0,
            continuous_shoulder_deadzone=1.0 / 3.0,
        )
    )

    first_state = adapter.decode(np.array([0.25, 0.5, 0.75], dtype=np.float32))
    second_state = adapter.decode(np.array([0.25, 0.5, 0.75], dtype=np.float32))

    assert first_state == ControllerState(
        joypad_mask=ACCELERATE_MASK | DRIFT_RIGHT_MASK,
        left_stick_x=0.25,
    )
    assert second_state == ControllerState(
        joypad_mask=ACCELERATE_MASK | DRIFT_RIGHT_MASK,
        left_stick_x=0.25,
    )


def test_hybrid_steer_drive_drift_adapter_uses_dict_action_space() -> None:
    adapter = HybridSteerDriveDriftActionAdapter(ActionConfig(name="hybrid_steer_drive_drift"))

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


def test_hybrid_steer_drive_drift_adapter_decodes_discrete_drift_branch() -> None:
    adapter = HybridSteerDriveDriftActionAdapter(
        ActionConfig(name="hybrid_steer_drive_drift", continuous_drive_deadzone=0.2)
    )

    control_state = adapter.decode(
        {
            "continuous": np.array([0.25, 0.7], dtype=np.float32),
            "discrete": np.array([1], dtype=np.int64),
        }
    )

    assert control_state == ControllerState(
        joypad_mask=ACCELERATE_MASK | DRIFT_LEFT_MASK,
        left_stick_x=0.25,
    )


def test_hybrid_steer_drive_boost_drift_adapter_decodes_boost_branch() -> None:
    adapter = HybridSteerDriveBoostDriftActionAdapter(
        ActionConfig(name="hybrid_steer_drive_boost_drift", continuous_drive_deadzone=0.2)
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
        joypad_mask=ACCELERATE_MASK | DRIFT_RIGHT_MASK | BOOST_MASK,
        left_stick_x=0.25,
    )


def test_hybrid_steer_drive_boost_drift_adapter_masks_boost_branch() -> None:
    adapter = HybridSteerDriveBoostDriftActionAdapter(
        ActionConfig(name="hybrid_steer_drive_boost_drift")
    )

    mask = adapter.action_mask(dynamic_overrides={"boost": (0,)})

    assert mask.tolist() == [True, True, True, True, False]


def test_hybrid_steer_drive_boost_shoulder_primitive_adapter_reserves_future_primitives() -> None:
    adapter = HybridSteerDriveBoostShoulderPrimitiveActionAdapter(
        ActionConfig(name="hybrid_steer_drive_boost_shoulder_primitive")
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


def test_hybrid_steer_drive_boost_shoulder_primitive_adapter_decodes_current_drift_values() -> None:
    adapter = HybridSteerDriveBoostShoulderPrimitiveActionAdapter(
        ActionConfig(
            name="hybrid_steer_drive_boost_shoulder_primitive",
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
        joypad_mask=ACCELERATE_MASK | DRIFT_RIGHT_MASK | BOOST_MASK,
        left_stick_x=0.25,
    )


def test_hybrid_steer_drive_boost_shoulder_primitive_adapter_keeps_reserved_values_noop() -> None:
    adapter = HybridSteerDriveBoostShoulderPrimitiveActionAdapter(
        ActionConfig(
            name="hybrid_steer_drive_boost_shoulder_primitive",
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


def test_hybrid_steer_drive_boost_shoulder_primitive_adapter_decodes_air_brake_axis() -> None:
    adapter = HybridSteerDriveBoostShoulderPrimitiveActionAdapter(
        ActionConfig(
            name="hybrid_steer_drive_boost_shoulder_primitive",
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


def test_hybrid_steer_drive_boost_shoulder_primitive_adapter_can_disable_air_brake() -> None:
    adapter = HybridSteerDriveBoostShoulderPrimitiveActionAdapter(
        ActionConfig(
            name="hybrid_steer_drive_boost_shoulder_primitive",
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


def test_hybrid_steer_drive_boost_shoulder_primitive_adapter_keeps_neutral_air_brake_off() -> None:
    adapter = HybridSteerDriveBoostShoulderPrimitiveActionAdapter(
        ActionConfig(
            name="hybrid_steer_drive_boost_shoulder_primitive",
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


def test_hybrid_steer_drive_boost_shoulder_primitive_adapter_allows_config_unmask() -> None:
    adapter = HybridSteerDriveBoostShoulderPrimitiveActionAdapter(
        ActionConfig(name="hybrid_steer_drive_boost_shoulder_primitive")
    )

    mask = adapter.action_mask(base_overrides={"shoulder": (0, 3), "boost": (0,)})

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

    assert isinstance(adapter, SteerDriveBoostDriftActionAdapter)


def test_extended_adapter_uses_four_head_multidiscrete_space() -> None:
    adapter = SteerDriveBoostDriftActionAdapter(
        ActionConfig(name="steer_drive_boost_drift", steer_buckets=7)
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


def test_extended_adapter_decodes_accelerate_boost_and_explicit_right_shoulder() -> None:
    adapter = SteerDriveBoostDriftActionAdapter(
        ActionConfig(name="steer_drive_boost_drift", steer_buckets=7)
    )

    control_state = adapter.decode(np.array([1, 1, 1, 2], dtype=np.int64))

    assert control_state.joypad_mask == (ACCELERATE_MASK | BOOST_MASK | DRIFT_RIGHT_MASK)
    assert control_state.left_stick_x == pytest.approx(-2.0 / 3.0)


def test_extended_adapter_decodes_explicit_left_shoulder_independent_of_steer() -> None:
    adapter = SteerDriveBoostDriftActionAdapter(
        ActionConfig(name="steer_drive_boost_drift", steer_buckets=7)
    )

    control_state = adapter.decode(np.array([5, 0, 0, 1], dtype=np.int64))

    assert control_state.joypad_mask == DRIFT_LEFT_MASK
    assert control_state.left_stick_x == pytest.approx(2.0 / 3.0)


def test_extended_adapter_decodes_explicit_right_shoulder_while_steering_straight() -> None:
    adapter = SteerDriveBoostDriftActionAdapter(
        ActionConfig(name="steer_drive_boost_drift", steer_buckets=7)
    )

    control_state = adapter.decode(np.array([3, 2, 0, 2], dtype=np.int64))

    assert control_state.joypad_mask == (AIR_BRAKE_MASK | DRIFT_RIGHT_MASK)
    assert control_state.left_stick_x == 0.0


def test_extended_adapter_rejects_wrong_action_shape() -> None:
    adapter = SteerDriveBoostDriftActionAdapter(
        ActionConfig(name="steer_drive_boost_drift", steer_buckets=7)
    )

    with pytest.raises(ValueError):
        adapter.decode(np.array([3, 1, 1], dtype=np.int64))


def test_extended_adapter_rejects_removed_both_shoulders_index() -> None:
    adapter = SteerDriveBoostDriftActionAdapter(
        ActionConfig(name="steer_drive_boost_drift", steer_buckets=7)
    )

    with pytest.raises(ValueError, match="Invalid shoulder index 3"):
        adapter.decode(np.array([3, 1, 1, 3], dtype=np.int64))


def test_build_action_adapter_supports_boost_only_variant() -> None:
    adapter = build_action_adapter(ActionConfig(name="steer_drive_boost"))

    assert isinstance(adapter, SteerDriveBoostActionAdapter)


def test_build_action_adapter_supports_current_shoulder_variants() -> None:
    assert isinstance(
        build_action_adapter(ActionConfig(name="steer_drive_boost_shoulder")),
        SteerDriveBoostShoulderActionAdapter,
    )
    assert isinstance(
        build_action_adapter(ActionConfig(name="steer_gas_air_brake_boost_shoulder")),
        SteerGasAirBrakeBoostShoulderActionAdapter,
    )
    assert isinstance(
        build_action_adapter(ActionConfig(name="continuous_steer_drive_shoulder")),
        ContinuousSteerDriveShoulderActionAdapter,
    )
    assert isinstance(
        build_action_adapter(ActionConfig(name="hybrid_steer_drive_shoulder")),
        HybridSteerDriveShoulderActionAdapter,
    )
    assert isinstance(
        build_action_adapter(ActionConfig(name="hybrid_steer_drive_boost_shoulder")),
        HybridSteerDriveBoostShoulderActionAdapter,
    )
    assert isinstance(
        build_action_adapter(ActionConfig(name="hybrid_steer_gas_air_brake_boost_shoulder")),
        HybridSteerGasAirBrakeBoostShoulderActionAdapter,
    )


def test_build_action_adapter_supports_steer_gas_air_brake_boost_drift_variant() -> None:
    adapter = build_action_adapter(ActionConfig(name="steer_gas_air_brake_boost_drift"))

    assert isinstance(adapter, SteerGasAirBrakeBoostDriftActionAdapter)


def test_build_action_adapter_supports_continuous_steer_drive_variant() -> None:
    adapter = build_action_adapter(ActionConfig(name="continuous_steer_drive"))

    assert isinstance(adapter, ContinuousSteerDriveActionAdapter)


def test_build_action_adapter_supports_continuous_steer_drive_drift_variant() -> None:
    adapter = build_action_adapter(ActionConfig(name="continuous_steer_drive_drift"))

    assert isinstance(adapter, ContinuousSteerDriveDriftActionAdapter)


def test_build_action_adapter_supports_hybrid_steer_drive_drift_variant() -> None:
    adapter = build_action_adapter(ActionConfig(name="hybrid_steer_drive_drift"))

    assert isinstance(adapter, HybridSteerDriveDriftActionAdapter)


def test_build_action_adapter_supports_hybrid_boost_drift_variant() -> None:
    adapter = build_action_adapter(ActionConfig(name="hybrid_steer_drive_boost_drift"))

    assert isinstance(adapter, HybridSteerDriveBoostDriftActionAdapter)


def test_build_action_adapter_supports_hybrid_boost_shoulder_primitive_variant() -> None:
    adapter = build_action_adapter(ActionConfig(name="hybrid_steer_drive_boost_shoulder_primitive"))

    assert isinstance(adapter, HybridSteerDriveBoostShoulderPrimitiveActionAdapter)


def test_action_adapter_registry_exposes_registered_names() -> None:
    assert action_adapter_names() == tuple(ACTION_ADAPTER_REGISTRY)


def test_extended_adapter_action_mask_defaults_to_all_actions_enabled() -> None:
    adapter = SteerDriveBoostDriftActionAdapter(
        ActionConfig(name="steer_drive_boost_drift", steer_buckets=7)
    )

    mask = adapter.action_mask()

    assert mask.dtype == np.bool_
    assert mask.tolist() == ([True] * (7 + 3 + 2 + 3))


def test_extended_adapter_action_mask_can_disable_shoulder_branch() -> None:
    base_mask = ActionMaskConfig(shoulder=(0,))
    adapter = SteerDriveBoostDriftActionAdapter(
        ActionConfig(
            name="steer_drive_boost_drift",
            steer_buckets=7,
            mask=base_mask,
        )
    )

    mask = adapter.action_mask(base_overrides=base_mask.branch_overrides())

    assert mask.tolist() == (([True] * 7) + ([True] * 3) + ([True] * 2) + [True, False, False])


def test_stage_action_mask_overrides_base_mask_for_same_branch() -> None:
    base_mask = ActionMaskConfig(shoulder=(0,))
    adapter = SteerDriveBoostDriftActionAdapter(
        ActionConfig(
            name="steer_drive_boost_drift",
            steer_buckets=7,
            mask=base_mask,
        )
    )

    mask = adapter.action_mask(
        base_overrides=base_mask.branch_overrides(),
        stage_overrides={"shoulder": (0, 1, 2)},
    )

    assert mask.tolist() == ([True] * (7 + 3 + 2 + 3))
