# src/rl_fzerox/core/envs/actions/continuous_steer_drive.py
from __future__ import annotations

import numpy as np
from gymnasium import spaces

from fzerox_emulator import ControllerState
from rl_fzerox.core.config.schema import ActionConfig
from rl_fzerox.core.envs.actions.base import ActionValue, DiscreteActionDimension
from rl_fzerox.core.envs.actions.steer_drive import BRAKE_MASK, THROTTLE_MASK

_ACTION_SIZE = 2


class ContinuousSteerDriveActionAdapter:
    """Map continuous steering and drive intent to simple held controls."""

    def __init__(self, config: ActionConfig) -> None:
        self._drive_deadzone = float(config.continuous_drive_deadzone)
        self._action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self._idle_action = np.zeros(_ACTION_SIZE, dtype=np.float32)

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
        """Translate one continuous action into a held steering/drive state."""

        steer, drive = _parse_continuous_pair(action)
        joypad_mask = 0
        if drive > self._drive_deadzone:
            joypad_mask = THROTTLE_MASK
        elif drive < -self._drive_deadzone:
            joypad_mask = BRAKE_MASK

        return ControllerState(
            joypad_mask=joypad_mask,
            left_stick_x=steer,
        )

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
    values = _continuous_action_array(action)
    steer = float(np.clip(values[0], -1.0, 1.0))
    drive = float(np.clip(values[1], -1.0, 1.0))
    return steer, drive


def _continuous_action_array(action: ActionValue) -> np.ndarray:
    if isinstance(action, np.ndarray):
        values = action.astype(np.float32, copy=False).reshape(-1)
    elif isinstance(action, int | float | np.integer | np.floating):
        values = np.array([float(action)], dtype=np.float32)
    elif isinstance(action, str | bytes):
        raise ValueError("Continuous steer-drive action must be numeric")
    else:
        values = np.array([float(value) for value in action], dtype=np.float32)

    if values.size != _ACTION_SIZE:
        raise ValueError(
            f"Continuous steer-drive actions must contain exactly {_ACTION_SIZE} values: "
            "[steer, drive]"
        )
    if not np.isfinite(values).all():
        raise ValueError("Continuous steer-drive action values must be finite")
    return values
