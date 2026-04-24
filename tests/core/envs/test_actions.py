# tests/core/envs/test_actions.py
import numpy as np
import pytest
from gymnasium.spaces import Box, Dict, MultiDiscrete

from fzerox_emulator import ControllerState
from rl_fzerox.core.config.schema import ActionConfig
from rl_fzerox.core.envs.actions import (
    ACCELERATE_MASK,
    AIR_BRAKE_MASK,
    BOOST_MASK,
    LEAN_RIGHT_MASK,
    ContinuousSteerDriveActionAdapter,
    HybridSteerDriveAirBrakeBoostLeanPitchActionAdapter,
    HybridSteerGasAirBrakeBoostLeanActionAdapter,
    HybridSteerGasBoostLeanActionAdapter,
    SteerDriveActionAdapter,
    SteerGasAirBrakeBoostLeanActionAdapter,
)
from rl_fzerox.core.envs.actions.continuous_controls import continuous_drive_gas_level


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


def test_steer_gas_air_brake_boost_lean_adapter_uses_full_discrete_heads() -> None:
    adapter = SteerGasAirBrakeBoostLeanActionAdapter(
        ActionConfig(name="steer_gas_air_brake_boost_lean", steer_buckets=3)
    )

    assert isinstance(adapter.action_space, MultiDiscrete)
    assert adapter.action_space.nvec.tolist() == [3, 2, 2, 2, 3]
    assert np.array_equal(adapter.idle_action, np.array([1, 0, 0, 0, 0], dtype=np.int64))
    assert adapter.action_mask().tolist() == [True] * (3 + 2 + 2 + 2 + 3)


def test_steer_gas_air_brake_boost_lean_adapter_decodes_parallel_buttons() -> None:
    adapter = SteerGasAirBrakeBoostLeanActionAdapter(
        ActionConfig(name="steer_gas_air_brake_boost_lean", steer_buckets=3)
    )

    control_state = adapter.decode(np.array([0, 1, 1, 1, 2], dtype=np.int64))

    assert control_state == ControllerState(
        joypad_mask=ACCELERATE_MASK | AIR_BRAKE_MASK | BOOST_MASK | LEAN_RIGHT_MASK,
        left_stick_x=-1.0,
    )


def test_steer_gas_air_brake_boost_lean_adapter_masks_branches_separately() -> None:
    adapter = SteerGasAirBrakeBoostLeanActionAdapter(
        ActionConfig(name="steer_gas_air_brake_boost_lean", steer_buckets=3)
    )

    mask = adapter.action_mask(
        base_overrides={
            "gas": (1,),
            "air_brake": (0,),
            "boost": (0,),
            "lean": (0,),
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


def test_hybrid_steer_gas_boost_lean_adapter_uses_one_continuous_axis() -> None:
    adapter = HybridSteerGasBoostLeanActionAdapter(ActionConfig(name="hybrid_steer_gas_boost_lean"))

    assert isinstance(adapter.action_space, Dict)
    assert isinstance(adapter.action_space.spaces["continuous"], Box)
    assert adapter.action_space.spaces["continuous"].shape == (1,)
    assert isinstance(adapter.action_space.spaces["discrete"], MultiDiscrete)
    assert adapter.action_space.spaces["discrete"].nvec.tolist() == [2, 2, 3]
    assert adapter.idle_action["continuous"].tolist() == [0.0]
    assert adapter.idle_action["discrete"].tolist() == [0, 0, 0]
    assert tuple(dimension.label for dimension in adapter.action_dimensions) == (
        "gas",
        "boost",
        "lean",
    )


def test_hybrid_steer_gas_boost_lean_adapter_decodes_discrete_buttons() -> None:
    adapter = HybridSteerGasBoostLeanActionAdapter(ActionConfig(name="hybrid_steer_gas_boost_lean"))

    control_state = adapter.decode(
        {
            "continuous": np.array([-0.75], dtype=np.float32),
            "discrete": np.array([1, 1, 2], dtype=np.int64),
        }
    )

    assert control_state == ControllerState(
        joypad_mask=ACCELERATE_MASK | BOOST_MASK | LEAN_RIGHT_MASK,
        left_stick_x=-0.75,
    )


def test_hybrid_steer_gas_boost_lean_adapter_masks_discrete_heads() -> None:
    adapter = HybridSteerGasBoostLeanActionAdapter(ActionConfig(name="hybrid_steer_gas_boost_lean"))

    mask = adapter.action_mask(
        base_overrides={
            "gas": (1,),
            "boost": (0,),
            "lean": (0,),
        },
    )

    assert mask.tolist() == [
        False,
        True,
        True,
        False,
        True,
        False,
        False,
    ]


def test_hybrid_steer_gas_air_brake_boost_lean_adapter_uses_one_continuous_axis() -> None:
    adapter = HybridSteerGasAirBrakeBoostLeanActionAdapter(
        ActionConfig(name="hybrid_steer_gas_air_brake_boost_lean")
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
        "lean",
    )


def test_hybrid_steer_gas_air_brake_boost_lean_adapter_decodes_discrete_buttons() -> None:
    adapter = HybridSteerGasAirBrakeBoostLeanActionAdapter(
        ActionConfig(name="hybrid_steer_gas_air_brake_boost_lean")
    )

    control_state = adapter.decode(
        {
            "continuous": np.array([-0.75], dtype=np.float32),
            "discrete": np.array([1, 1, 1, 2], dtype=np.int64),
        }
    )

    assert control_state == ControllerState(
        joypad_mask=ACCELERATE_MASK | AIR_BRAKE_MASK | BOOST_MASK | LEAN_RIGHT_MASK,
        left_stick_x=-0.75,
    )


def test_hybrid_steer_gas_air_brake_boost_lean_adapter_applies_steer_curve() -> None:
    adapter = HybridSteerGasAirBrakeBoostLeanActionAdapter(
        ActionConfig(
            name="hybrid_steer_gas_air_brake_boost_lean",
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


def test_hybrid_steer_gas_air_brake_boost_lean_adapter_masks_discrete_heads() -> None:
    adapter = HybridSteerGasAirBrakeBoostLeanActionAdapter(
        ActionConfig(name="hybrid_steer_gas_air_brake_boost_lean")
    )

    mask = adapter.action_mask(
        base_overrides={
            "gas": (1,),
            "air_brake": (0,),
            "boost": (0,),
            "lean": (0,),
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


def test_hybrid_steer_drive_air_brake_boost_lean_pitch_adapter_uses_expected_space() -> None:
    adapter = HybridSteerDriveAirBrakeBoostLeanPitchActionAdapter(
        ActionConfig(name="hybrid_steer_drive_air_brake_boost_lean_pitch")
    )

    assert isinstance(adapter.action_space, Dict)
    assert isinstance(adapter.action_space.spaces["continuous"], Box)
    assert adapter.action_space.spaces["continuous"].shape == (2,)
    assert isinstance(adapter.action_space.spaces["discrete"], MultiDiscrete)
    assert adapter.action_space.spaces["discrete"].nvec.tolist() == [2, 2, 3, 5]
    assert adapter.idle_action["continuous"].tolist() == [0.0, 0.0]
    assert adapter.idle_action["discrete"].tolist() == [0, 0, 0, 2]
    assert tuple(dimension.label for dimension in adapter.action_dimensions) == (
        "air_brake",
        "boost",
        "lean",
        "pitch",
    )


def test_hybrid_steer_drive_air_brake_boost_lean_pitch_adapter_decodes_pitch() -> None:
    adapter = HybridSteerDriveAirBrakeBoostLeanPitchActionAdapter(
        ActionConfig(name="hybrid_steer_drive_air_brake_boost_lean_pitch")
    )

    control_state = adapter.decode(
        {
            "continuous": np.array([-0.25, 1.0], dtype=np.float32),
            "discrete": np.array([1, 1, 2, 4], dtype=np.int64),
        }
    )

    assert control_state == ControllerState(
        joypad_mask=ACCELERATE_MASK | AIR_BRAKE_MASK | BOOST_MASK | LEAN_RIGHT_MASK,
        left_stick_x=-0.25,
        left_stick_y=1.0,
    )


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


def test_continuous_drive_pwm_supports_deadzone_and_full_zone() -> None:
    assert (
        continuous_drive_gas_level(
            -0.95,
            mode="pwm",
            deadzone=0.10,
            full_threshold=0.80,
        )
        == 0.0
    )
    assert continuous_drive_gas_level(
        -0.45,
        mode="pwm",
        deadzone=0.10,
        full_threshold=0.80,
    ) == pytest.approx((0.55 - 0.10) / (0.80 - 0.10))
    assert (
        continuous_drive_gas_level(
            -0.20,
            mode="pwm",
            deadzone=0.10,
            full_threshold=0.80,
        )
        == 1.0
    )


def test_continuous_drive_pwm_applies_min_level_inside_deadzone() -> None:
    assert continuous_drive_gas_level(
        -0.95,
        mode="pwm",
        deadzone=0.10,
        full_threshold=0.80,
        min_level=0.25,
    ) == pytest.approx(0.25)
    assert continuous_drive_gas_level(
        -0.45,
        mode="pwm",
        deadzone=0.10,
        full_threshold=0.80,
        min_level=0.25,
    ) == pytest.approx(0.25 + (0.75 * ((0.55 - 0.10) / (0.80 - 0.10))))
    assert (
        continuous_drive_gas_level(
            -0.20,
            mode="pwm",
            deadzone=0.10,
            full_threshold=0.80,
            min_level=0.25,
        )
        == 1.0
    )


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
