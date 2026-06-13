# tests/core/envs/test_actions.py
import numpy as np
import pytest
from gymnasium.spaces import Box, Dict, MultiDiscrete

from fzerox_emulator import RaceControlState
from rl_fzerox.core.envs.actions import (
    RACE_CONTROL_MASKS,
    ConfiguredDiscreteActionAdapter,
    ConfiguredHybridActionAdapter,
    ResettableActionAdapter,
    build_action_adapter,
)
from rl_fzerox.core.envs.actions.continuous_controls import (
    action_drive_axis,
    continuous_drive_gas_level,
    requested_gas_level,
)
from rl_fzerox.core.runtime_spec.schema import ActionConfig
from tests.support.action_configs import (
    configured_discrete_action,
    configured_hybrid_action,
)


def test_configured_discrete_action_space_matches_gas_air_brake_layout() -> None:
    adapter = ConfiguredDiscreteActionAdapter(
        configured_discrete_action("steer", "gas", "air_brake")
    )

    assert isinstance(adapter.action_space, MultiDiscrete)
    assert adapter.action_space.nvec.tolist() == [7, 2, 2]
    assert np.array_equal(adapter.idle_action, np.array([3, 0, 0], dtype=np.int64))


def test_configured_discrete_action_space_matches_full_button_layout() -> None:
    adapter = ConfiguredDiscreteActionAdapter(
        configured_discrete_action(
            "steer",
            "gas",
            "air_brake",
            "boost",
            "lean",
            steer_buckets=3,
        )
    )

    assert isinstance(adapter.action_space, MultiDiscrete)
    assert adapter.action_space.nvec.tolist() == [3, 2, 2, 2, 3]
    assert np.array_equal(adapter.idle_action, np.array([1, 0, 0, 0, 0], dtype=np.int64))


def test_action_config_rejects_even_steer_bucket_counts() -> None:
    with pytest.raises(
        ValueError,
        match="action bucket counts must be odd so one bucket maps to neutral",
    ):
        ActionConfig(steer_buckets=6)


def test_action_config_rejects_non_positive_steer_response_power() -> None:
    with pytest.raises(ValueError):
        ActionConfig(steer_response_power=0.0)


def test_action_config_rejects_spin_without_three_way_lean() -> None:
    with pytest.raises(ValueError, match="spin axis requires three_way categorical lean"):
        configured_discrete_action(
            "lean",
            "spin",
            lean_output_mode="four_way_categorical",
        )


def test_configured_discrete_gas_air_brake_layout_decodes_buttons() -> None:
    adapter = ConfiguredDiscreteActionAdapter(
        configured_discrete_action("steer", "gas", "air_brake")
    )

    accelerate_state = adapter.decode(np.array([3, 1, 0], dtype=np.int64))
    air_brake_state = adapter.decode(np.array([3, 0, 1], dtype=np.int64))

    assert accelerate_state == RaceControlState(gas=True, stick_x=0.0)
    assert air_brake_state == RaceControlState(air_brake=True, stick_x=0.0)


def test_configured_discrete_applies_steer_curve() -> None:
    adapter = ConfiguredDiscreteActionAdapter(
        configured_discrete_action("steer", "gas", steer_buckets=5, steer_response_power=0.5)
    )

    control_state = adapter.decode(np.array([1, 0], dtype=np.int64))

    assert control_state.stick_x == pytest.approx(-(0.5**0.5))


def test_configured_discrete_pitch_ignores_deadzone_for_controller_output() -> None:
    adapter = ConfiguredDiscreteActionAdapter(
        configured_discrete_action("pitch", pitch_buckets=5, pitch_deadzone=0.6)
    )

    half_pitch = adapter.decode(np.array([1], dtype=np.int64))
    full_pitch = adapter.decode(np.array([0], dtype=np.int64))

    assert half_pitch.pitch == -0.5
    assert full_pitch.pitch == -1.0


def test_configured_discrete_rejects_wrong_action_shape() -> None:
    adapter = ConfiguredDiscreteActionAdapter(configured_discrete_action("steer", "gas"))

    with pytest.raises(ValueError):
        adapter.decode(np.array([3], dtype=np.int64))


def test_configured_discrete_full_button_layout_decodes_parallel_buttons() -> None:
    adapter = ConfiguredDiscreteActionAdapter(
        configured_discrete_action("steer", "gas", "air_brake", "boost", "lean", steer_buckets=3)
    )

    control_state = adapter.decode(np.array([0, 1, 1, 1, 2], dtype=np.int64))

    assert control_state == RaceControlState(
        gas=True,
        air_brake=True,
        boost=True,
        lean_right=True,
        stick_x=-1.0,
    )


def test_configured_discrete_spin_branch_decodes_macro_request_only() -> None:
    adapter = ConfiguredDiscreteActionAdapter(configured_discrete_action("lean", "spin"))

    decoded = adapter.decode_request(np.array([0, 2], dtype=np.int64))

    assert decoded.control_state == RaceControlState()
    assert decoded.spin_request == "right"


def test_configured_discrete_four_way_lean_decodes_both_buttons() -> None:
    adapter = ConfiguredDiscreteActionAdapter(
        configured_discrete_action(
            "lean",
            lean_output_mode="four_way_categorical",
        )
    )

    control_state = adapter.decode(np.array([3], dtype=np.int64))

    assert control_state.control_mask == (
        RACE_CONTROL_MASKS.lean_left | RACE_CONTROL_MASKS.lean_right
    )


def test_configured_discrete_independent_lean_decodes_binary_buttons() -> None:
    adapter = ConfiguredDiscreteActionAdapter(
        configured_discrete_action(
            "lean_left",
            "lean_right",
            lean_output_mode="independent_buttons",
        )
    )

    control_state = adapter.decode(np.array([1, 1], dtype=np.int64))

    assert control_state.control_mask == (
        RACE_CONTROL_MASKS.lean_left | RACE_CONTROL_MASKS.lean_right
    )


def test_configured_discrete_masks_branches_separately() -> None:
    adapter = ConfiguredDiscreteActionAdapter(
        configured_discrete_action("steer", "gas", "air_brake", "boost", "lean", steer_buckets=3)
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


def test_configured_hybrid_steer_gas_boost_lean_uses_expected_space() -> None:
    adapter = ConfiguredHybridActionAdapter(
        configured_hybrid_action(
            continuous_axes=("steer",),
            discrete_axes=("gas", "boost", "lean"),
        )
    )

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


def test_configured_hybrid_steer_gas_boost_lean_decodes_discrete_buttons() -> None:
    adapter = ConfiguredHybridActionAdapter(
        configured_hybrid_action(
            continuous_axes=("steer",),
            discrete_axes=("gas", "boost", "lean"),
        )
    )

    control_state = adapter.decode(
        {
            "continuous": np.array([-0.75], dtype=np.float32),
            "discrete": np.array([1, 1, 2], dtype=np.int64),
        }
    )

    assert control_state == RaceControlState(
        gas=True,
        boost=True,
        lean_right=True,
        stick_x=-0.75,
    )


def test_configured_hybrid_spin_branch_decodes_macro_request() -> None:
    adapter = ConfiguredHybridActionAdapter(
        configured_hybrid_action(
            continuous_axes=("steer",),
            discrete_axes=("lean", "spin"),
        )
    )

    decoded = adapter.decode_request(
        {
            "continuous": np.array([0.25], dtype=np.float32),
            "discrete": np.array([1, 1], dtype=np.int64),
        }
    )

    assert decoded.control_state == RaceControlState(lean_left=True, stick_x=0.25)
    assert decoded.spin_request == "left"


def test_configured_hybrid_pitch_ignores_deadzone_for_controller_output() -> None:
    adapter = ConfiguredHybridActionAdapter(
        configured_hybrid_action(
            continuous_axes=("pitch",),
            pitch_deadzone=0.25,
        )
    )

    small_pitch = adapter.decode(
        {
            "continuous": np.array([0.2], dtype=np.float32),
            "discrete": np.array([], dtype=np.int64),
        }
    )
    live_pitch = adapter.decode(
        {
            "continuous": np.array([0.3], dtype=np.float32),
            "discrete": np.array([], dtype=np.int64),
        }
    )

    assert small_pitch.pitch == pytest.approx(0.2)
    assert live_pitch.pitch == pytest.approx(0.3)


def test_configured_hybrid_steer_gas_air_brake_boost_lean_masks_heads() -> None:
    adapter = ConfiguredHybridActionAdapter(
        configured_hybrid_action(
            continuous_axes=("steer",),
            discrete_axes=("gas", "air_brake", "boost", "lean"),
        )
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


def test_action_drive_axis_is_none_when_layout_has_no_drive_axis() -> None:
    config = configured_hybrid_action(
        continuous_axes=("steer", "pitch"),
        discrete_axes=("gas",),
    )
    runtime = config.runtime()
    action = {
        "continuous": np.array([0.1, 0.6], dtype=np.float32),
        "discrete": np.array([1], dtype=np.int64),
    }

    drive_axis = action_drive_axis(
        action,
        build_action_adapter(config).action_space,
        drive_axis_index=runtime.continuous_drive_axis_index(),
    )

    assert drive_axis is None


def test_requested_gas_level_uses_discrete_accelerate_when_no_drive_axis_exists() -> None:
    gas_level = requested_gas_level(
        control_state=RaceControlState(gas=True, pitch=0.6),
        drive_axis=None,
        continuous_drive_deadzone=0.05,
        continuous_drive_full_threshold=0.85,
        continuous_drive_min_thrust=0.25,
    )

    assert gas_level == 1.0


def test_configured_hybrid_decodes_continuous_air_brake_lane() -> None:
    adapter = build_action_adapter(
        configured_hybrid_action(
            continuous_axes=("steer", "air_brake"),
            discrete_axes=("gas", "boost"),
            continuous_air_brake_deadzone=0.0,
            continuous_air_brake_full_threshold=1.0,
            continuous_air_brake_min_duty=0.0,
        )
    )

    control_state = adapter.decode(
        {
            "continuous": np.array([0.25, 1.0], dtype=np.float32),
            "discrete": np.array([0, 1], dtype=np.int64),
        }
    )

    assert control_state == RaceControlState(air_brake=True, boost=True, stick_x=0.25)


def test_configured_hybrid_steer_drive_action_space_uses_dict_branches() -> None:
    adapter = build_action_adapter(
        configured_hybrid_action(
            continuous_axes=("steer", "drive"),
            discrete_axes=(),
        )
    )

    assert isinstance(adapter, ConfiguredHybridActionAdapter)
    assert isinstance(adapter.action_space, Dict)
    assert adapter.action_space.spaces["continuous"].shape == (2,)
    discrete_space = adapter.action_space.spaces["discrete"]
    assert isinstance(discrete_space, MultiDiscrete)
    assert discrete_space.nvec.tolist() == []


def test_configured_hybrid_steer_drive_decodes_accelerate_and_coast() -> None:
    adapter = build_action_adapter(
        configured_hybrid_action(
            continuous_axes=("steer", "drive"),
            discrete_axes=(),
            continuous_drive_deadzone=0.2,
        )
    )

    accelerate_state = adapter.decode(
        {
            "continuous": np.array([0.5, 0.7], dtype=np.float32),
            "discrete": np.array([], dtype=np.int64),
        }
    )
    coast_state = adapter.decode(
        {
            "continuous": np.array([-0.5, -0.7], dtype=np.float32),
            "discrete": np.array([], dtype=np.int64),
        }
    )

    assert accelerate_state == RaceControlState(gas=True, stick_x=0.5)
    assert coast_state == RaceControlState(stick_x=-0.5)


def test_configured_hybrid_drive_pwm_defaults_to_full_throttle() -> None:
    adapter = build_action_adapter(
        configured_hybrid_action(
            continuous_axes=("steer", "drive"),
            discrete_axes=(),
            continuous_drive_deadzone=0.0,
            continuous_drive_full_threshold=1.0,
            continuous_drive_min_thrust=0.0,
        )
    )
    assert isinstance(adapter, ResettableActionAdapter)

    coast_mask = adapter.decode(
        {
            "continuous": np.array([0.0, -1.0], dtype=np.float32),
            "discrete": np.array([], dtype=np.int64),
        }
    ).control_mask
    partial_masks = [
        adapter.decode(
            {
                "continuous": np.array([0.0, -0.5], dtype=np.float32),
                "discrete": np.array([], dtype=np.int64),
            }
        ).control_mask
        for _ in range(4)
    ]
    adapter.reset()
    mid_masks = [
        adapter.decode(
            {
                "continuous": np.array([0.0, 0.0], dtype=np.float32),
                "discrete": np.array([], dtype=np.int64),
            }
        ).control_mask
        for _ in range(4)
    ]
    adapter.reset()
    full_masks = [
        adapter.decode(
            {
                "continuous": np.array([0.0, 1.0], dtype=np.float32),
                "discrete": np.array([], dtype=np.int64),
            }
        ).control_mask
        for _ in range(3)
    ]

    assert coast_mask == 0
    assert partial_masks == [0, 0, 0, RACE_CONTROL_MASKS.accelerate]
    assert mid_masks == [0, RACE_CONTROL_MASKS.accelerate, 0, RACE_CONTROL_MASKS.accelerate]
    assert full_masks == [
        RACE_CONTROL_MASKS.accelerate,
        RACE_CONTROL_MASKS.accelerate,
        RACE_CONTROL_MASKS.accelerate,
    ]


def test_continuous_drive_curve_supports_deadzone_and_full_zone() -> None:
    assert (
        continuous_drive_gas_level(
            -0.95,
            deadzone=0.10,
            full_threshold=0.80,
        )
        == 0.0
    )
    assert continuous_drive_gas_level(
        -0.45,
        deadzone=0.10,
        full_threshold=0.80,
    ) == pytest.approx((0.275 - 0.10) / (0.80 - 0.10))
    assert (
        continuous_drive_gas_level(
            0.60,
            deadzone=0.10,
            full_threshold=0.80,
        )
        == 1.0
    )


def test_continuous_drive_curve_applies_min_thrust_inside_deadzone() -> None:
    assert continuous_drive_gas_level(
        -0.95,
        deadzone=0.10,
        full_threshold=0.80,
        min_thrust=0.25,
    ) == pytest.approx(0.25)
    assert continuous_drive_gas_level(
        -0.45,
        deadzone=0.10,
        full_threshold=0.80,
        min_thrust=0.25,
    ) == pytest.approx(0.25 + (0.75 * ((0.275 - 0.10) / (0.80 - 0.10))))
    assert (
        continuous_drive_gas_level(
            0.60,
            deadzone=0.10,
            full_threshold=0.80,
            min_thrust=0.25,
        )
        == 1.0
    )


def test_configured_hybrid_can_force_full_thrust() -> None:
    adapter = build_action_adapter(
        configured_hybrid_action(
            continuous_axes=("steer", "drive"),
            discrete_axes=(),
            continuous_drive_min_thrust=1.0,
        )
    )

    control_state = adapter.decode(
        {
            "continuous": np.array([-0.5, -1.0], dtype=np.float32),
            "discrete": np.array([], dtype=np.int64),
        }
    )

    assert control_state == RaceControlState(gas=True, stick_x=-0.5)
