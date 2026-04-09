# tests/core/envs/test_actions.py
import numpy as np
import pytest
from gymnasium.spaces import MultiDiscrete

from fzerox_emulator import ControllerState
from rl_fzerox.core.config.schema import ActionConfig, ActionMaskConfig
from rl_fzerox.core.envs.actions import (
    ACTION_ADAPTER_REGISTRY,
    BOOST_MASK,
    BRAKE_MASK,
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


def test_steer_drive_adapter_decodes_center_throttle_action() -> None:
    adapter = SteerDriveActionAdapter(ActionConfig(name="steer_drive", steer_buckets=7))

    control_state = adapter.decode(np.array([3, 1], dtype=np.int64))

    assert control_state == ControllerState(
        joypad_mask=THROTTLE_MASK,
        left_stick_x=0.0,
    )


def test_steer_drive_adapter_decodes_center_brake_action() -> None:
    adapter = SteerDriveActionAdapter(ActionConfig(name="steer_drive", steer_buckets=7))

    control_state = adapter.decode(np.array([3, 2], dtype=np.int64))

    assert control_state == ControllerState(
        joypad_mask=BRAKE_MASK,
        left_stick_x=0.0,
    )


def test_steer_drive_adapter_rejects_wrong_action_shape() -> None:
    adapter = SteerDriveActionAdapter(ActionConfig(name="steer_drive", steer_buckets=7))

    with pytest.raises(ValueError):
        adapter.decode(np.array([3], dtype=np.int64))


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


def test_boost_adapter_decodes_throttle_and_boost() -> None:
    adapter = SteerDriveBoostActionAdapter(ActionConfig(name="steer_drive_boost", steer_buckets=7))

    control_state = adapter.decode(np.array([4, 1, 1], dtype=np.int64))

    assert control_state.joypad_mask == (THROTTLE_MASK | BOOST_MASK)
    assert control_state.left_stick_x == pytest.approx(1.0 / 3.0)


def test_boost_adapter_decodes_brake_without_boost() -> None:
    adapter = SteerDriveBoostActionAdapter(ActionConfig(name="steer_drive_boost", steer_buckets=7))

    control_state = adapter.decode(np.array([3, 2, 0], dtype=np.int64))

    assert control_state.joypad_mask == BRAKE_MASK
    assert control_state.left_stick_x == 0.0


def test_boost_adapter_rejects_wrong_action_shape() -> None:
    adapter = SteerDriveBoostActionAdapter(ActionConfig(name="steer_drive_boost", steer_buckets=7))

    with pytest.raises(ValueError):
        adapter.decode(np.array([3, 1], dtype=np.int64))


def test_extended_adapter_decodes_throttle_boost_and_explicit_right_shoulder() -> None:
    adapter = SteerDriveBoostDriftActionAdapter(
        ActionConfig(name="steer_drive_boost_drift", steer_buckets=7)
    )

    control_state = adapter.decode(np.array([1, 1, 1, 2], dtype=np.int64))

    assert control_state.joypad_mask == (THROTTLE_MASK | BOOST_MASK | DRIFT_RIGHT_MASK)
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

    assert control_state.joypad_mask == (BRAKE_MASK | DRIFT_RIGHT_MASK)
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
