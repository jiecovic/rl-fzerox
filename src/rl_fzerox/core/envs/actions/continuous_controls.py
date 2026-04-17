# src/rl_fzerox/core/envs/actions/continuous_controls.py
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from fzerox_emulator.arrays import ContinuousAction, DiscreteAction
from rl_fzerox.core.envs.actions.base import ActionBranchValue, ActionValue
from rl_fzerox.core.envs.actions.steer_drive import ACCELERATE_MASK


class ContinuousDriveDecoder:
    """Decode one continuous drive axis into the game's binary accelerate button."""

    def __init__(self, *, mode: str, deadzone: float) -> None:
        self._mode = mode
        self._deadzone = deadzone
        self._pwm_phase = 0.0

    def reset(self) -> None:
        """Clear deterministic PWM phase so episodes start from neutral drive."""

        self._pwm_phase = 0.0

    def decode(self, drive: float) -> int:
        if self._mode == "always_accelerate":
            return ACCELERATE_MASK
        if self._mode == "threshold":
            return _threshold_drive_mask(drive, deadzone=self._deadzone)
        if self._mode == "pwm":
            return self._pwm_drive_mask(_pwm_duty_cycle(drive, deadzone=self._deadzone))
        raise ValueError(f"Unsupported continuous drive mode: {self._mode!r}")

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

    def __init__(self, *, deadzone: float) -> None:
        self._deadzone = deadzone
        self._pwm_phase = 0.0

    def reset(self) -> None:
        """Clear deterministic PWM phase so episodes start from neutral button state."""

        self._pwm_phase = 0.0

    def decode(self, value: float, *, button_mask: int) -> int:
        duty = _positive_button_pwm_duty_cycle(value, deadzone=self._deadzone)
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
    if isinstance(action, np.ndarray):
        values = action.astype(np.float32, copy=False).reshape(-1)
    elif isinstance(action, int | float | np.integer | np.floating):
        values = np.array([float(action)], dtype=np.float32)
    elif isinstance(action, str | bytes):
        raise ValueError("Continuous steer-drive action must be numeric")
    else:
        values = np.array([float(value) for value in action], dtype=np.float32)

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
    if isinstance(action, np.ndarray):
        values = action.astype(np.int64, copy=False).reshape(-1)
    elif isinstance(action, int | float | np.integer | np.floating):
        values = np.array([int(action)], dtype=np.int64)
    elif isinstance(action, str | bytes):
        raise ValueError(f"{action_label} action must be numeric")
    else:
        values = np.array([int(value) for value in action], dtype=np.int64)

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


def _threshold_drive_mask(drive: float, *, deadzone: float) -> int:
    # The SAC experiment keeps air brake out of the action space: negative drive coasts.
    return ACCELERATE_MASK if drive > deadzone else 0


def _pwm_duty_cycle(drive: float, *, deadzone: float) -> float:
    # F-Zero X usually wants full gas: negative values reduce duty, 0+ holds full throttle.
    duty = min(max(drive + 1.0, 0.0), 1.0)
    if duty <= deadzone:
        return 0.0
    return (duty - deadzone) / (1.0 - deadzone)


def _positive_button_pwm_duty_cycle(value: float, *, deadzone: float) -> float:
    # One-sided buttons should be neutral at 0; positive values express duty.
    if value <= deadzone:
        return 0.0
    return (value - deadzone) / (1.0 - deadzone)
