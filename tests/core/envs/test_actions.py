# tests/core/envs/test_actions.py
import numpy as np
import pytest
from gymnasium.spaces import MultiDiscrete

from rl_fzerox.core.config.schema import ActionConfig
from rl_fzerox.core.emulator.control import ControllerState
from rl_fzerox.core.envs.actions import (
    ACTION_ADAPTER_REGISTRY,
    BOOST_MASK,
    DRIFT_LEFT_MASK,
    DRIFT_RIGHT_MASK,
    THROTTLE_MASK,
    SteerDriveActionAdapter,
    SteerDriveBoostActionAdapter,
    SteerDriveBoostDriftActionAdapter,
    action_adapter_names,
    build_action_adapter,
)


def test_steer_drive_adapter_uses_default_seven_bucket_multidiscrete_space() -> None:
    adapter = SteerDriveActionAdapter(ActionConfig())

    assert isinstance(adapter.action_space, MultiDiscrete)
    assert adapter.action_space.nvec.tolist() == [7, 2]
    assert np.array_equal(adapter.idle_action, np.array([3, 0], dtype=np.int64))


def test_steer_drive_adapter_supports_custom_bucket_counts() -> None:
    adapter = SteerDriveActionAdapter(ActionConfig(steer_buckets=7))

    assert adapter.action_space.nvec.tolist() == [7, 2]
    assert np.array_equal(adapter.idle_action, np.array([3, 0], dtype=np.int64))


def test_steer_drive_adapter_decodes_center_throttle_action() -> None:
    adapter = SteerDriveActionAdapter(ActionConfig(steer_buckets=7))

    control_state = adapter.decode(np.array([3, 1], dtype=np.int64))

    assert control_state == ControllerState(
        joypad_mask=THROTTLE_MASK,
        left_stick_x=0.0,
    )


def test_steer_drive_adapter_rejects_wrong_action_shape() -> None:
    adapter = SteerDriveActionAdapter(ActionConfig(steer_buckets=7))

    with pytest.raises(ValueError):
        adapter.decode(np.array([3], dtype=np.int64))


def test_build_action_adapter_uses_basic_adapter_by_default() -> None:
    adapter = build_action_adapter(ActionConfig())

    assert isinstance(adapter, SteerDriveActionAdapter)


def test_extended_adapter_uses_four_head_multidiscrete_space() -> None:
    adapter = SteerDriveBoostDriftActionAdapter(
        ActionConfig(name="steer_drive_boost_drift", steer_buckets=7)
    )

    assert isinstance(adapter.action_space, MultiDiscrete)
    assert adapter.action_space.nvec.tolist() == [7, 2, 2, 3]
    assert np.array_equal(adapter.idle_action, np.array([3, 0, 0, 0], dtype=np.int64))


def test_boost_adapter_uses_three_head_multidiscrete_space() -> None:
    adapter = SteerDriveBoostActionAdapter(
        ActionConfig(name="steer_drive_boost", steer_buckets=7)
    )

    assert isinstance(adapter.action_space, MultiDiscrete)
    assert adapter.action_space.nvec.tolist() == [7, 2, 2]
    assert np.array_equal(adapter.idle_action, np.array([3, 0, 0], dtype=np.int64))


def test_boost_adapter_decodes_throttle_and_boost() -> None:
    adapter = SteerDriveBoostActionAdapter(
        ActionConfig(name="steer_drive_boost", steer_buckets=7)
    )

    control_state = adapter.decode(np.array([4, 1, 1], dtype=np.int64))

    assert control_state.joypad_mask == (THROTTLE_MASK | BOOST_MASK)
    assert control_state.left_stick_x == pytest.approx(1.0 / 3.0)


def test_boost_adapter_rejects_wrong_action_shape() -> None:
    adapter = SteerDriveBoostActionAdapter(
        ActionConfig(name="steer_drive_boost", steer_buckets=7)
    )

    with pytest.raises(ValueError):
        adapter.decode(np.array([3, 1], dtype=np.int64))


def test_extended_adapter_decodes_throttle_boost_and_left_drift() -> None:
    adapter = SteerDriveBoostDriftActionAdapter(
        ActionConfig(name="steer_drive_boost_drift", steer_buckets=7)
    )

    control_state = adapter.decode(np.array([4, 1, 1, 1], dtype=np.int64))

    assert control_state.joypad_mask == (THROTTLE_MASK | BOOST_MASK | DRIFT_LEFT_MASK)
    assert control_state.left_stick_x == pytest.approx(1.0 / 3.0)


def test_extended_adapter_decodes_right_drift_without_boost() -> None:
    adapter = SteerDriveBoostDriftActionAdapter(
        ActionConfig(name="steer_drive_boost_drift", steer_buckets=7)
    )

    control_state = adapter.decode(np.array([1, 0, 0, 2], dtype=np.int64))

    assert control_state.joypad_mask == DRIFT_RIGHT_MASK
    assert control_state.left_stick_x == pytest.approx(-2.0 / 3.0)


def test_extended_adapter_rejects_wrong_action_shape() -> None:
    adapter = SteerDriveBoostDriftActionAdapter(
        ActionConfig(name="steer_drive_boost_drift", steer_buckets=7)
    )

    with pytest.raises(ValueError):
        adapter.decode(np.array([3, 1, 1], dtype=np.int64))


def test_build_action_adapter_supports_boost_only_variant() -> None:
    adapter = build_action_adapter(ActionConfig(name="steer_drive_boost"))

    assert isinstance(adapter, SteerDriveBoostActionAdapter)


def test_action_adapter_registry_exposes_registered_names() -> None:
    assert action_adapter_names() == tuple(ACTION_ADAPTER_REGISTRY)
