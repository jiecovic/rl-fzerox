# tests/ui/test_viewer_controls.py
import numpy as np
import pytest

from fzerox_emulator import ControllerState
from rl_fzerox.core.envs.actions import (
    ACCELERATE_MASK,
    AIR_BRAKE_MASK,
    BOOST_MASK,
    LEAN_LEFT_MASK,
)
from rl_fzerox.ui.watch.runtime.snapshots import (
    BOOST_LAMP_CONFIG,
    _next_boost_lamp_level,
)
from rl_fzerox.ui.watch.view.panels.viz import _control_viz


def test_control_viz_includes_visualized_control_state() -> None:
    control_viz = _control_viz(
        ControllerState(
            joypad_mask=ACCELERATE_MASK | BOOST_MASK | LEAN_LEFT_MASK,
            left_stick_x=0.5,
        ),
        gas_level=1.0,
        policy_action=None,
    )

    assert control_viz.gas_level == 1.0
    assert control_viz.steer_x == 0.5
    assert control_viz.boost_pressed
    assert control_viz.boost_lamp_level == 1.0
    assert control_viz.lean_direction == -1


def test_control_viz_lights_boost_lamp_from_boost_active() -> None:
    control_viz = _control_viz(
        ControllerState(),
        gas_level=1.0,
        policy_action=None,
        boost_active=True,
    )

    assert not control_viz.boost_pressed
    assert control_viz.boost_active
    assert control_viz.boost_lamp_level == 0.55


def test_boost_lamp_flashes_then_fades_to_active_and_off_levels() -> None:
    lamp_level = _next_boost_lamp_level(
        previous=0.0,
        control_state=ControllerState(joypad_mask=BOOST_MASK),
        boost_active=False,
        action_repeat=1,
    )
    assert lamp_level == BOOST_LAMP_CONFIG.manual_level

    lamp_level = _next_boost_lamp_level(
        previous=lamp_level,
        control_state=ControllerState(),
        boost_active=True,
        action_repeat=1,
    )
    assert BOOST_LAMP_CONFIG.active_level < lamp_level < BOOST_LAMP_CONFIG.manual_level

    for _ in range(20):
        lamp_level = _next_boost_lamp_level(
            previous=lamp_level,
            control_state=ControllerState(),
            boost_active=True,
            action_repeat=1,
        )
    assert lamp_level == pytest.approx(BOOST_LAMP_CONFIG.active_level)

    lamp_level = _next_boost_lamp_level(
        previous=lamp_level,
        control_state=ControllerState(),
        boost_active=False,
        action_repeat=1,
    )
    assert 0.0 < lamp_level < BOOST_LAMP_CONFIG.active_level


def test_control_viz_visualizes_air_brake_button_without_air_brake_axis() -> None:
    control_viz = _control_viz(
        ControllerState(joypad_mask=AIR_BRAKE_MASK, left_stick_x=0.0),
        gas_level=0.0,
        policy_action=None,
    )

    assert control_viz.gas_level == 0
    assert control_viz.air_brake_axis == pytest.approx(1.0)


def test_control_viz_keeps_gas_unipolar_with_air_brake_button_pressed() -> None:
    control_viz = _control_viz(
        ControllerState(joypad_mask=ACCELERATE_MASK | AIR_BRAKE_MASK),
        gas_level=1.0,
        policy_action=np.array([1, 1, 1], dtype=np.int64),
    )

    assert control_viz.gas_level == 1.0
    assert control_viz.air_brake_axis == pytest.approx(1.0)


def test_control_viz_visualizes_canonical_gas_level() -> None:
    control_viz = _control_viz(
        ControllerState(left_stick_x=0.25),
        gas_level=0.5,
        policy_action=np.array([0.25, -0.5], dtype=np.float32),
        continuous_drive_deadzone=0.0,
    )

    assert control_viz.gas_level == pytest.approx(0.5)
    assert control_viz.air_brake_axis is None


def test_control_viz_visualizes_pitch_axis() -> None:
    control_viz = _control_viz(
        ControllerState(left_stick_y=-0.5),
        gas_level=0.0,
        policy_action=None,
    )

    assert control_viz.pitch_y == pytest.approx(-0.5)


def test_control_viz_visualizes_thrust_warning_threshold() -> None:
    control_viz = _control_viz(
        ControllerState(),
        gas_level=0.25,
        thrust_warning_threshold=0.5,
        policy_action=None,
    )

    assert control_viz.gas_level == pytest.approx(0.25)
    assert control_viz.thrust_warning_threshold == pytest.approx(0.5)


def test_control_viz_visualizes_forced_full_accelerate_drive_mode() -> None:
    control_viz = _control_viz(
        ControllerState(joypad_mask=ACCELERATE_MASK, left_stick_x=0.25),
        gas_level=1.0,
        policy_action={
            "continuous": np.array([0.25, -1.0, 0.5], dtype=np.float32),
            "discrete": np.array([0, 0], dtype=np.int64),
        },
        continuous_drive_deadzone=0.0,
    )

    assert control_viz.gas_level == pytest.approx(1.0)
    assert control_viz.air_brake_axis == pytest.approx(0.5)


def test_control_viz_visualizes_hybrid_canonical_gas_level() -> None:
    control_viz = _control_viz(
        ControllerState(
            joypad_mask=ACCELERATE_MASK | LEAN_LEFT_MASK,
            left_stick_x=0.25,
        ),
        gas_level=1.0,
        policy_action={
            "continuous": np.array([0.25, 0.5], dtype=np.float32),
            "discrete": np.array([1], dtype=np.int64),
        },
        continuous_drive_deadzone=0.0,
    )

    assert control_viz.gas_level == pytest.approx(1.0)
    assert control_viz.air_brake_axis is None
    assert control_viz.lean_direction == -1


def test_control_viz_visualizes_hybrid_policy_air_brake_axis() -> None:
    control_viz = _control_viz(
        ControllerState(joypad_mask=ACCELERATE_MASK | AIR_BRAKE_MASK),
        gas_level=1.0,
        policy_action={
            "continuous": np.array([0.0, 0.5, 0.5], dtype=np.float32),
            "discrete": np.array([0, 0], dtype=np.int64),
        },
        continuous_drive_deadzone=0.0,
    )

    assert control_viz.gas_level == pytest.approx(1.0)
    assert control_viz.air_brake_axis == pytest.approx(0.5)


def test_control_viz_hides_disabled_hybrid_policy_air_brake_axis() -> None:
    control_viz = _control_viz(
        ControllerState(joypad_mask=ACCELERATE_MASK),
        gas_level=1.0,
        policy_action={
            "continuous": np.array([0.0, 0.5, 0.5], dtype=np.float32),
            "discrete": np.array([0, 0], dtype=np.int64),
        },
        continuous_drive_deadzone=0.0,
        continuous_air_brake_mode="off",
    )

    assert control_viz.gas_level == pytest.approx(1.0)
    assert control_viz.air_brake_axis is None


def test_control_viz_grays_disabled_ground_air_brake_axis() -> None:
    control_viz = _control_viz(
        ControllerState(joypad_mask=ACCELERATE_MASK),
        gas_level=1.0,
        policy_action={
            "continuous": np.array([0.0, 0.5, 0.5], dtype=np.float32),
            "discrete": np.array([0, 0], dtype=np.int64),
        },
        continuous_drive_deadzone=0.0,
        continuous_air_brake_mode="disable_on_ground",
        continuous_air_brake_disabled=True,
    )

    assert control_viz.air_brake_axis == pytest.approx(0.5)
    assert control_viz.air_brake_disabled is True


def test_control_viz_uses_canonical_gas_level_for_discrete_policy_action() -> None:
    control_viz = _control_viz(
        ControllerState(joypad_mask=ACCELERATE_MASK, left_stick_x=0.0),
        gas_level=1.0,
        policy_action=np.array([4, 1], dtype=np.int64),
    )

    assert control_viz.gas_level == pytest.approx(1.0)
    assert control_viz.air_brake_axis is None
