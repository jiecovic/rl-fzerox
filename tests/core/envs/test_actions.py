# tests/core/envs/test_actions.py
import numpy as np
import pytest
from gymnasium.spaces import MultiDiscrete

from rl_fzerox.core.config.schema import ActionConfig
from rl_fzerox.core.emulator.control import ControllerState
from rl_fzerox.core.envs.actions import THROTTLE_MASK, SteerDriveActionAdapter


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
