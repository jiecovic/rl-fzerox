# src/rl_fzerox/core/envs/actions/continuous_steer_drive.py
from __future__ import annotations

import numpy as np
from gymnasium import spaces

from fzerox_emulator import ControllerState
from rl_fzerox.core.config.schema import ActionConfig
from rl_fzerox.core.envs.actions.base import ActionValue, DiscreteActionDimension
from rl_fzerox.core.envs.actions.steer_drive import THROTTLE_MASK
from rl_fzerox.core.envs.actions.steer_drive_boost_drift import (
    DRIFT_LEFT_MASK,
    DRIFT_RIGHT_MASK,
)

_STEER_DRIVE_ACTION_SIZE = 2
_STEER_DRIVE_DRIFT_ACTION_SIZE = 3


class ContinuousSteerDriveActionAdapter:
    """Map continuous steering and throttle intent to simple held controls."""

    def __init__(self, config: ActionConfig) -> None:
        self._drive_decoder = _ContinuousDriveDecoder(
            mode=config.continuous_drive_mode,
            deadzone=float(config.continuous_drive_deadzone),
        )
        self._action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self._idle_action = np.zeros(_STEER_DRIVE_ACTION_SIZE, dtype=np.float32)

    @property
    def action_space(self) -> spaces.Box:
        """Return the current policy action space."""

        return self._action_space

    @property
    def idle_action(self) -> np.ndarray:
        """Return the neutral action value for this action space."""

        return np.array(self._idle_action, copy=True)

    @property
    def action_dimensions(self) -> tuple[DiscreteActionDimension, ...]:
        """Return no discrete heads; this adapter is not maskable."""

        return ()

    def decode(self, action: ActionValue) -> ControllerState:
        """Translate one continuous action into a held steering/throttle state."""

        steer, drive = _parse_continuous_pair(action)
        joypad_mask = self._drive_decoder.decode(drive)

        return ControllerState(
            joypad_mask=joypad_mask,
            left_stick_x=steer,
        )

    def reset(self) -> None:
        """Reset the drive PWM accumulator for a new episode."""

        self._drive_decoder.reset()

    def action_mask(
        self,
        *,
        base_overrides: dict[str, tuple[int, ...]] | None = None,
        stage_overrides: dict[str, tuple[int, ...]] | None = None,
        dynamic_overrides: dict[str, tuple[int, ...]] | None = None,
    ) -> np.ndarray:
        """Return an empty mask so watch can call action_masks() safely."""

        _ = (base_overrides, stage_overrides, dynamic_overrides)
        return np.ones(0, dtype=bool)


class ContinuousSteerDriveDriftActionAdapter:
    """Map continuous steer/drive/drift intent to held controller inputs."""

    def __init__(self, config: ActionConfig) -> None:
        self._drive_decoder = _ContinuousDriveDecoder(
            mode=config.continuous_drive_mode,
            deadzone=float(config.continuous_drive_deadzone),
        )
        self._drift_deadzone = float(config.continuous_drift_deadzone)
        self._action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self._idle_action = np.zeros(_STEER_DRIVE_DRIFT_ACTION_SIZE, dtype=np.float32)

    @property
    def action_space(self) -> spaces.Box:
        """Return the current policy action space."""

        return self._action_space

    @property
    def idle_action(self) -> np.ndarray:
        """Return the neutral action value for this action space."""

        return np.array(self._idle_action, copy=True)

    @property
    def action_dimensions(self) -> tuple[DiscreteActionDimension, ...]:
        """Return no discrete heads; this adapter is not maskable."""

        return ()

    def decode(self, action: ActionValue) -> ControllerState:
        """Translate one continuous action into steering, drive, and drift."""

        steer, drive, drift = _parse_continuous_triplet(action)
        joypad_mask = self._drive_decoder.decode(drive)
        if drift > self._drift_deadzone:
            joypad_mask |= DRIFT_RIGHT_MASK
        elif drift < -self._drift_deadzone:
            joypad_mask |= DRIFT_LEFT_MASK

        return ControllerState(
            joypad_mask=joypad_mask,
            left_stick_x=steer,
        )

    def reset(self) -> None:
        """Reset the drive PWM accumulator for a new episode."""

        self._drive_decoder.reset()

    def action_mask(
        self,
        *,
        base_overrides: dict[str, tuple[int, ...]] | None = None,
        stage_overrides: dict[str, tuple[int, ...]] | None = None,
        dynamic_overrides: dict[str, tuple[int, ...]] | None = None,
    ) -> np.ndarray:
        """Return an empty mask so watch can call action_masks() safely."""

        _ = (base_overrides, stage_overrides, dynamic_overrides)
        return np.ones(0, dtype=bool)


def _parse_continuous_pair(action: ActionValue) -> tuple[float, float]:
    values = _continuous_action_array(
        action,
        expected_size=_STEER_DRIVE_ACTION_SIZE,
        action_label="Continuous steer-drive",
        field_labels=("steer", "drive"),
    )
    steer = float(np.clip(values[0], -1.0, 1.0))
    drive = float(np.clip(values[1], -1.0, 1.0))
    return steer, drive


def _parse_continuous_triplet(action: ActionValue) -> tuple[float, float, float]:
    values = _continuous_action_array(
        action,
        expected_size=_STEER_DRIVE_DRIFT_ACTION_SIZE,
        action_label="Continuous steer-drive-drift",
        field_labels=("steer", "drive", "drift"),
    )
    steer = float(np.clip(values[0], -1.0, 1.0))
    drive = float(np.clip(values[1], -1.0, 1.0))
    drift = float(np.clip(values[2], -1.0, 1.0))
    return steer, drive, drift


class _ContinuousDriveDecoder:
    """Decode one continuous drive axis into the game's binary throttle button."""

    def __init__(self, *, mode: str, deadzone: float) -> None:
        self._mode = mode
        self._deadzone = deadzone
        self._pwm_phase = 0.0

    def reset(self) -> None:
        """Clear deterministic PWM phase so episodes start from neutral drive."""

        self._pwm_phase = 0.0

    def decode(self, drive: float) -> int:
        if self._mode == "threshold":
            return _threshold_drive_mask(drive, deadzone=self._deadzone)
        if self._mode == "pwm":
            return self._pwm_drive_mask(drive)
        raise ValueError(f"Unsupported continuous drive mode: {self._mode!r}")

    def _pwm_drive_mask(self, drive: float) -> int:
        duty = _pwm_duty_cycle(drive, deadzone=self._deadzone)
        if duty <= 0.0:
            self._pwm_phase = 0.0
            return 0

        self._pwm_phase += duty
        if self._pwm_phase < 1.0:
            return 0

        self._pwm_phase -= 1.0
        return THROTTLE_MASK


def _threshold_drive_mask(drive: float, *, deadzone: float) -> int:
    # The SAC experiment keeps brake out of the action space: negative drive coasts.
    return THROTTLE_MASK if drive > deadzone else 0


def _pwm_duty_cycle(drive: float, *, deadzone: float) -> float:
    if drive <= deadzone:
        return 0.0
    return (drive - deadzone) / (1.0 - deadzone)


def _continuous_action_array(
    action: ActionValue,
    *,
    expected_size: int,
    action_label: str,
    field_labels: tuple[str, ...],
) -> np.ndarray:
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
