# src/rl_fzerox/core/envs/actions/continuous_controls.py
from __future__ import annotations

from collections.abc import Mapping
from typing import TypeGuard

import numpy as np
from gymnasium import spaces

from fzerox_emulator import ControllerState
from fzerox_emulator.arrays import ContinuousAction, DiscreteAction
from rl_fzerox.core.domain.hybrid_action import HYBRID_CONTINUOUS_ACTION_KEY
from rl_fzerox.core.envs.actions.base import ActionBranchValue, ActionValue
from rl_fzerox.core.envs.actions.buttons import ACCELERATE_MASK


class ContinuousDriveDecoder:
    """Decode one continuous drive axis into the game's binary accelerate button."""

    def __init__(
        self,
        *,
        deadzone: float,
        full_threshold: float = 1.0,
        min_thrust: float = 0.0,
    ) -> None:
        self._deadzone = deadzone
        self._full_threshold = full_threshold
        self._min_thrust = min_thrust
        self._pwm_phase = 0.0

    def reset(self) -> None:
        """Clear deterministic PWM phase so episodes start from neutral drive."""

        self._pwm_phase = 0.0

    def decode(self, drive: float) -> int:
        return self._pwm_drive_mask(
            continuous_drive_gas_level(
                drive,
                deadzone=self._deadzone,
                full_threshold=self._full_threshold,
                min_thrust=self._min_thrust,
            )
        )

    def _pwm_drive_mask(self, duty: float) -> int:
        if duty <= 0.0:
            self._pwm_phase = 0.0
            return 0

        self._pwm_phase += duty
        if self._pwm_phase < 1.0:
            return 0

        self._pwm_phase -= 1.0
        return ACCELERATE_MASK


class ContinuousButtonPwmDecoder:
    """Decode positive continuous intent into a binary button with deterministic PWM."""

    def __init__(
        self,
        *,
        deadzone: float,
        full_threshold: float = 1.0,
        min_duty: float = 0.0,
    ) -> None:
        self._deadzone = deadzone
        self._full_threshold = full_threshold
        self._min_duty = min_duty
        self._pwm_phase = 0.0

    def reset(self) -> None:
        """Clear deterministic PWM phase so episodes start from neutral button state."""

        self._pwm_phase = 0.0

    def decode(self, value: float, *, button_mask: int) -> int:
        duty = _positive_button_pwm_duty_cycle(
            value,
            deadzone=self._deadzone,
            full_threshold=self._full_threshold,
            min_duty=self._min_duty,
        )
        if duty <= 0.0:
            self._pwm_phase = 0.0
            return 0

        self._pwm_phase += duty
        if self._pwm_phase < 1.0:
            return 0

        self._pwm_phase -= 1.0
        return button_mask


def continuous_action_array(
    action: ActionValue,
    *,
    expected_size: int,
    action_label: str,
    field_labels: tuple[str, ...],
) -> ContinuousAction:
    if isinstance(action, Mapping):
        raise ValueError("Continuous steer-drive action must be a numeric sequence")
    values = _coerce_continuous_action_values(action)

    if values.size != expected_size:
        labels = ", ".join(field_labels)
        raise ValueError(
            f"{action_label} actions must contain exactly {expected_size} values: [{labels}]"
        )
    if not np.isfinite(values).all():
        raise ValueError(f"{action_label} action values must be finite")
    return values


def discrete_action_array(
    action: ActionBranchValue,
    *,
    expected_size: int,
    action_label: str,
    field_labels: tuple[str, ...],
) -> DiscreteAction:
    values = _coerce_discrete_action_values(action, action_label=action_label)

    if values.size != expected_size:
        labels = ", ".join(field_labels)
        raise ValueError(
            f"{action_label} actions must contain exactly {expected_size} values: [{labels}]"
        )
    return values


def hybrid_branch(
    action: Mapping[str, ActionBranchValue],
    branch_name: str,
) -> ActionBranchValue:
    try:
        return action[branch_name]
    except KeyError as exc:
        raise ValueError(f"Hybrid action is missing {branch_name!r} branch") from exc


def _coerce_continuous_action_values(action: ActionValue) -> ContinuousAction:
    if isinstance(action, np.ndarray):
        return np.asarray(action, dtype=np.float32).reshape(-1)
    if _is_action_scalar(action):
        return np.array([float(action)], dtype=np.float32)
    if isinstance(action, str | bytes):
        raise ValueError("Continuous steer-drive action must be numeric")
    if not isinstance(action, tuple | list):
        raise ValueError("Continuous steer-drive action must be numeric")
    return np.array([float(value) for value in action], dtype=np.float32)


def _coerce_discrete_action_values(
    action: ActionBranchValue,
    *,
    action_label: str,
) -> DiscreteAction:
    if isinstance(action, np.ndarray):
        return np.asarray(action, dtype=np.int64).reshape(-1)
    if _is_action_scalar(action):
        return np.array([int(action)], dtype=np.int64)
    if isinstance(action, str | bytes):
        raise ValueError(f"{action_label} action must be numeric")
    if not isinstance(action, tuple | list):
        raise ValueError(f"{action_label} action must be numeric")
    return np.array([int(value) for value in action], dtype=np.int64)


def _is_action_scalar(value: object) -> TypeGuard[int | float | np.integer | np.floating]:
    return isinstance(value, int | float | np.integer | np.floating)


def continuous_drive_gas_level(
    drive: float,
    *,
    deadzone: float,
    full_threshold: float = 1.0,
    min_thrust: float = 0.0,
) -> float:
    """Return normalized thrust intent for one continuous drive axis."""

    return _continuous_drive_thrust_curve(
        drive,
        deadzone=deadzone,
        full_threshold=full_threshold,
        min_thrust=min_thrust,
    )


def requested_gas_level(
    *,
    control_state: ControllerState,
    drive_axis: float | None,
    continuous_drive_deadzone: float,
    continuous_drive_full_threshold: float = 1.0,
    continuous_drive_min_thrust: float = 0.0,
) -> float:
    """Return one canonical 0..1 gas intent for reward and HUD consumers."""

    if drive_axis is not None:
        return continuous_drive_gas_level(
            drive_axis,
            deadzone=continuous_drive_deadzone,
            full_threshold=continuous_drive_full_threshold,
            min_thrust=continuous_drive_min_thrust,
        )
    return 1.0 if control_state.joypad_mask & ACCELERATE_MASK else 0.0


def action_drive_axis(
    action: ActionValue,
    action_space: spaces.Space,
    *,
    drive_axis_index: int | None = None,
) -> float | None:
    """Extract the raw continuous drive axis when the action space exposes one."""

    if drive_axis_index is None:
        return None

    source: object
    if isinstance(action_space, spaces.Dict):
        if not isinstance(action, Mapping):
            return None
        source = action.get(HYBRID_CONTINUOUS_ACTION_KEY)
    elif isinstance(action_space, spaces.Box):
        source = action
    else:
        return None
    if source is None or isinstance(source, str | bytes):
        return None
    try:
        values = np.asarray(source, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return None
    axis_index = int(drive_axis_index)
    if axis_index < 0 or values.size <= axis_index or not np.isfinite(values[axis_index]):
        return None
    return float(np.clip(values[axis_index], -1.0, 1.0))


def _continuous_drive_thrust_curve(
    drive: float,
    *,
    deadzone: float,
    full_threshold: float,
    min_thrust: float,
) -> float:
    # Map the centered policy output evenly into a 0..1 thrust duty.
    duty = min(max((drive + 1.0) / 2.0, 0.0), 1.0)
    minimum = _clamp_unit(min_thrust)
    if duty <= deadzone or bool(np.isclose(duty, deadzone)):
        return minimum
    if duty >= full_threshold or bool(np.isclose(duty, full_threshold)):
        return 1.0
    scaled = (duty - deadzone) / (full_threshold - deadzone)
    return minimum + ((1.0 - minimum) * scaled)


def _clamp_unit(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


def _positive_button_pwm_duty_cycle(
    value: float,
    *,
    deadzone: float,
    full_threshold: float,
    min_duty: float,
) -> float:
    # One-sided buttons should be neutral at 0; positive values express duty.
    if value <= deadzone:
        return 0.0
    minimum = _clamp_unit(min_duty)
    if value >= full_threshold or bool(np.isclose(value, full_threshold)):
        return 1.0
    scaled = (value - deadzone) / (full_threshold - deadzone)
    return minimum + ((1.0 - minimum) * scaled)
