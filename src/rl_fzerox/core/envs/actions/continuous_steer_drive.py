# src/rl_fzerox/core/envs/actions/continuous_steer_drive.py
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from gymnasium import spaces

from fzerox_emulator import ControllerState
from rl_fzerox.core.config.schema import ActionConfig
from rl_fzerox.core.envs.actions.base import (
    ActionBranchValue,
    ActionValue,
    DiscreteActionDimension,
    build_flat_action_mask,
)
from rl_fzerox.core.envs.actions.steer_drive import BRAKE_MASK, THROTTLE_MASK
from rl_fzerox.core.envs.actions.steer_drive_boost_drift import (
    BOOST_MASK,
    DRIFT_LEFT_MASK,
    DRIFT_RIGHT_MASK,
)

_STEER_DRIVE_ACTION_SIZE = 2
_STEER_DRIVE_DRIFT_ACTION_SIZE = 3
_HYBRID_STEER_DRIVE_CONTINUOUS_SIZE = 2
_HYBRID_STEER_DRIVE_BRAKE_CONTINUOUS_SIZE = 3
_HYBRID_DRIFT_DISCRETE_SIZE = 1
_HYBRID_BOOST_DRIFT_DISCRETE_SIZE = 2
_HYBRID_SHOULDER_PRIMITIVE_SIZE = 7
_HYBRID_CONTINUOUS_KEY = "continuous"
_HYBRID_DISCRETE_KEY = "discrete"
_HYBRID_ACTION_DIMENSIONS = (DiscreteActionDimension("shoulder", 3),)
_HYBRID_BOOST_ACTION_DIMENSIONS = (
    DiscreteActionDimension("shoulder", 3),
    DiscreteActionDimension("boost", 2),
)
_HYBRID_SHOULDER_PRIMITIVE_ACTION_DIMENSIONS = (
    DiscreteActionDimension("shoulder", _HYBRID_SHOULDER_PRIMITIVE_SIZE),
    DiscreteActionDimension("boost", 2),
)
_HYBRID_ENABLED_SHOULDER_PRIMITIVES = (0, 1, 2)


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


class HybridSteerDriveDriftActionAdapter:
    """Map hybrid PPO actions to continuous steer/drive and discrete drift."""

    def __init__(self, config: ActionConfig) -> None:
        self._drive_decoder = _ContinuousDriveDecoder(
            mode=config.continuous_drive_mode,
            deadzone=float(config.continuous_drive_deadzone),
        )
        self._action_space = spaces.Dict(
            {
                _HYBRID_CONTINUOUS_KEY: spaces.Box(
                    low=np.array([-1.0, -1.0], dtype=np.float32),
                    high=np.array([1.0, 1.0], dtype=np.float32),
                    dtype=np.float32,
                ),
                _HYBRID_DISCRETE_KEY: spaces.MultiDiscrete([3]),
            }
        )

    @property
    def action_space(self) -> spaces.Dict:
        """Return the public hybrid action space expected by HybridActionPPO."""

        return self._action_space

    @property
    def idle_action(self) -> dict[str, np.ndarray]:
        """Return a neutral hybrid action with no drift held."""

        return {
            _HYBRID_CONTINUOUS_KEY: np.zeros(
                _HYBRID_STEER_DRIVE_CONTINUOUS_SIZE,
                dtype=np.float32,
            ),
            _HYBRID_DISCRETE_KEY: np.zeros(_HYBRID_DRIFT_DISCRETE_SIZE, dtype=np.int64),
        }

    @property
    def action_dimensions(self) -> tuple[DiscreteActionDimension, ...]:
        """Return the discrete drift branch maskable by hybrid PPO."""

        return _HYBRID_ACTION_DIMENSIONS

    def decode(self, action: ActionValue) -> ControllerState:
        """Translate one hybrid action into steering, drive, and drift buttons."""

        steer, drive, drift = _parse_hybrid_steer_drive_drift(action)
        joypad_mask = self._drive_decoder.decode(drive)
        if drift == 1:
            joypad_mask |= DRIFT_LEFT_MASK
        elif drift == 2:
            joypad_mask |= DRIFT_RIGHT_MASK

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
        """Return the discrete-branch mask used by MaskableHybridActionPPO."""

        return build_flat_action_mask(
            self.action_dimensions,
            base_overrides=base_overrides,
            stage_overrides=stage_overrides,
            dynamic_overrides=dynamic_overrides,
        )


class HybridSteerDriveBoostDriftActionAdapter:
    """Map hybrid PPO actions to continuous steer/drive, drift, and boost."""

    def __init__(self, config: ActionConfig) -> None:
        self._drive_decoder = _ContinuousDriveDecoder(
            mode=config.continuous_drive_mode,
            deadzone=float(config.continuous_drive_deadzone),
        )
        self._action_space = spaces.Dict(
            {
                _HYBRID_CONTINUOUS_KEY: spaces.Box(
                    low=np.array([-1.0, -1.0], dtype=np.float32),
                    high=np.array([1.0, 1.0], dtype=np.float32),
                    dtype=np.float32,
                ),
                # Discrete order is shoulder, then boost, matching action_dimensions below.
                _HYBRID_DISCRETE_KEY: spaces.MultiDiscrete([3, 2]),
            }
        )

    @property
    def action_space(self) -> spaces.Dict:
        """Return the public hybrid action space expected by HybridActionPPO."""

        return self._action_space

    @property
    def idle_action(self) -> dict[str, np.ndarray]:
        """Return a neutral hybrid action with no drift or boost held."""

        return {
            _HYBRID_CONTINUOUS_KEY: np.zeros(
                _HYBRID_STEER_DRIVE_CONTINUOUS_SIZE,
                dtype=np.float32,
            ),
            _HYBRID_DISCRETE_KEY: np.zeros(
                _HYBRID_BOOST_DRIFT_DISCRETE_SIZE,
                dtype=np.int64,
            ),
        }

    @property
    def action_dimensions(self) -> tuple[DiscreteActionDimension, ...]:
        """Return the discrete branches maskable by maskable hybrid PPO."""

        return _HYBRID_BOOST_ACTION_DIMENSIONS

    def decode(self, action: ActionValue) -> ControllerState:
        """Translate one hybrid action into steering, drive, drift, and boost."""

        steer, drive, shoulder, boost = _parse_hybrid_steer_drive_boost_drift(action)
        joypad_mask = self._drive_decoder.decode(drive)
        if shoulder == 1:
            joypad_mask |= DRIFT_LEFT_MASK
        elif shoulder == 2:
            joypad_mask |= DRIFT_RIGHT_MASK
        if boost == 1:
            joypad_mask |= BOOST_MASK

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
        """Return the discrete-branch mask used by MaskableHybridActionPPO."""

        return build_flat_action_mask(
            self.action_dimensions,
            base_overrides=base_overrides,
            stage_overrides=stage_overrides,
            dynamic_overrides=dynamic_overrides,
        )


class HybridSteerDriveBoostShoulderPrimitiveActionAdapter:
    """Map hybrid PPO actions to continuous steer/gas/brake and shoulder primitives.

    The first three shoulder values match the current drift primitive:
    `off`, `drift_left`, `drift_right`. Values 3..6 reserve architecture
    capacity for future side-attack and spin primitives and currently decode
    as no-ops while masked out by default.
    """

    def __init__(self, config: ActionConfig) -> None:
        self._drive_decoder = _ContinuousDriveDecoder(
            mode=config.continuous_drive_mode,
            deadzone=float(config.continuous_drive_deadzone),
        )
        self._brake_decoder = _ContinuousButtonPwmDecoder(
            deadzone=float(config.continuous_drive_deadzone),
        )
        self._action_space = spaces.Dict(
            {
                _HYBRID_CONTINUOUS_KEY: spaces.Box(
                    low=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
                    high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
                    dtype=np.float32,
                ),
                # Discrete order is shoulder primitive, then boost.
                _HYBRID_DISCRETE_KEY: spaces.MultiDiscrete(
                    [_HYBRID_SHOULDER_PRIMITIVE_SIZE, 2]
                ),
            }
        )

    @property
    def action_space(self) -> spaces.Dict:
        """Return the public hybrid action space expected by HybridActionPPO."""

        return self._action_space

    @property
    def idle_action(self) -> dict[str, np.ndarray]:
        """Return a neutral hybrid action with no shoulder primitive or boost."""

        return {
            _HYBRID_CONTINUOUS_KEY: np.zeros(
                _HYBRID_STEER_DRIVE_BRAKE_CONTINUOUS_SIZE,
                dtype=np.float32,
            ),
            _HYBRID_DISCRETE_KEY: np.zeros(
                _HYBRID_BOOST_DRIFT_DISCRETE_SIZE,
                dtype=np.int64,
            ),
        }

    @property
    def action_dimensions(self) -> tuple[DiscreteActionDimension, ...]:
        """Return the discrete branches maskable by maskable hybrid PPO."""

        return _HYBRID_SHOULDER_PRIMITIVE_ACTION_DIMENSIONS

    def decode(self, action: ActionValue) -> ControllerState:
        """Translate one hybrid action into steering, drive, shoulder, and boost."""

        steer, drive, brake, shoulder, boost = _parse_hybrid_steer_drive_boost_shoulder(
            action
        )
        joypad_mask = self._drive_decoder.decode(drive) | self._brake_decoder.decode(
            brake,
            button_mask=BRAKE_MASK,
        )
        if shoulder == 1:
            joypad_mask |= DRIFT_LEFT_MASK
        elif shoulder == 2:
            joypad_mask |= DRIFT_RIGHT_MASK
        if boost == 1:
            joypad_mask |= BOOST_MASK

        return ControllerState(
            joypad_mask=joypad_mask,
            left_stick_x=steer,
        )

    def reset(self) -> None:
        """Reset drive and brake PWM accumulators for a new episode."""

        self._drive_decoder.reset()
        self._brake_decoder.reset()

    def action_mask(
        self,
        *,
        base_overrides: dict[str, tuple[int, ...]] | None = None,
        stage_overrides: dict[str, tuple[int, ...]] | None = None,
        dynamic_overrides: dict[str, tuple[int, ...]] | None = None,
    ) -> np.ndarray:
        """Return a mask that reserves future shoulder primitives by default."""

        return build_flat_action_mask(
            self.action_dimensions,
            base_overrides=_with_default_shoulder_primitive_mask(base_overrides),
            stage_overrides=stage_overrides,
            dynamic_overrides=dynamic_overrides,
        )


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


def _parse_hybrid_steer_drive_drift(action: ActionValue) -> tuple[float, float, int]:
    continuous_values, discrete_values = _parse_hybrid_branches(
        action,
        expected_size=_HYBRID_DRIFT_DISCRETE_SIZE,
        action_label="Hybrid steer-drive-drift discrete",
        field_labels=("drift",),
    )
    drift = int(discrete_values[0])
    if not 0 <= drift < 3:
        raise ValueError(f"Invalid hybrid drift index {drift}")

    steer = float(np.clip(continuous_values[0], -1.0, 1.0))
    drive = float(np.clip(continuous_values[1], -1.0, 1.0))
    return steer, drive, drift


def _parse_hybrid_steer_drive_boost_drift(
    action: ActionValue,
) -> tuple[float, float, int, int]:
    continuous_values, discrete_values = _parse_hybrid_branches(
        action,
        expected_size=_HYBRID_BOOST_DRIFT_DISCRETE_SIZE,
        action_label="Hybrid steer-drive-boost-drift discrete",
        field_labels=("shoulder", "boost"),
    )
    shoulder = int(discrete_values[0])
    boost = int(discrete_values[1])
    if not 0 <= shoulder < 3:
        raise ValueError(f"Invalid hybrid shoulder index {shoulder}")
    if not 0 <= boost < 2:
        raise ValueError(f"Invalid hybrid boost index {boost}")

    steer = float(np.clip(continuous_values[0], -1.0, 1.0))
    drive = float(np.clip(continuous_values[1], -1.0, 1.0))
    return steer, drive, shoulder, boost


def _parse_hybrid_steer_drive_boost_shoulder(
    action: ActionValue,
) -> tuple[float, float, float, int, int]:
    continuous_values, discrete_values = _parse_hybrid_branches(
        action,
        continuous_size=_HYBRID_STEER_DRIVE_BRAKE_CONTINUOUS_SIZE,
        continuous_field_labels=("steer", "drive", "brake"),
        expected_size=_HYBRID_BOOST_DRIFT_DISCRETE_SIZE,
        action_label="Hybrid steer-drive-boost-shoulder discrete",
        field_labels=("shoulder", "boost"),
    )
    shoulder = int(discrete_values[0])
    boost = int(discrete_values[1])
    if not 0 <= shoulder < _HYBRID_SHOULDER_PRIMITIVE_SIZE:
        raise ValueError(f"Invalid hybrid shoulder index {shoulder}")
    if not 0 <= boost < 2:
        raise ValueError(f"Invalid hybrid boost index {boost}")

    steer = float(np.clip(continuous_values[0], -1.0, 1.0))
    drive = float(np.clip(continuous_values[1], -1.0, 1.0))
    brake = float(np.clip(continuous_values[2], -1.0, 1.0))
    return steer, drive, brake, shoulder, boost


def _with_default_shoulder_primitive_mask(
    base_overrides: dict[str, tuple[int, ...]] | None,
) -> dict[str, tuple[int, ...]]:
    if base_overrides is not None and "shoulder" in base_overrides:
        return base_overrides
    overrides = dict(base_overrides or {})
    overrides["shoulder"] = _HYBRID_ENABLED_SHOULDER_PRIMITIVES
    return overrides


def _parse_hybrid_branches(
    action: ActionValue,
    *,
    continuous_size: int = _HYBRID_STEER_DRIVE_CONTINUOUS_SIZE,
    continuous_field_labels: tuple[str, ...] = ("steer", "drive"),
    expected_size: int,
    action_label: str,
    field_labels: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(action, Mapping):
        raise ValueError(
            "Hybrid steer-drive actions must be a mapping with "
            "'continuous' and 'discrete' branches"
        )

    continuous_values = _continuous_action_array(
        _hybrid_branch(action, _HYBRID_CONTINUOUS_KEY),
        expected_size=continuous_size,
        action_label="Hybrid steer-drive continuous",
        field_labels=continuous_field_labels,
    )
    discrete_values = _discrete_action_array(
        _hybrid_branch(action, _HYBRID_DISCRETE_KEY),
        expected_size=expected_size,
        action_label=action_label,
        field_labels=field_labels,
    )
    return continuous_values, discrete_values


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
            return self._pwm_drive_mask(
                _pwm_duty_cycle(drive, deadzone=self._deadzone)
            )
        raise ValueError(f"Unsupported continuous drive mode: {self._mode!r}")

    def _pwm_drive_mask(self, duty: float) -> int:
        if duty <= 0.0:
            self._pwm_phase = 0.0
            return 0

        self._pwm_phase += duty
        if self._pwm_phase < 1.0:
            return 0

        self._pwm_phase -= 1.0
        return THROTTLE_MASK


class _ContinuousButtonPwmDecoder:
    """Decode one continuous axis into a binary button using deterministic PWM."""

    def __init__(self, *, deadzone: float) -> None:
        self._deadzone = deadzone
        self._pwm_phase = 0.0

    def reset(self) -> None:
        """Clear deterministic PWM phase so episodes start from neutral button state."""

        self._pwm_phase = 0.0

    def decode(self, value: float, *, button_mask: int) -> int:
        duty = _pwm_duty_cycle(value, deadzone=self._deadzone)
        if duty <= 0.0:
            self._pwm_phase = 0.0
            return 0

        self._pwm_phase += duty
        if self._pwm_phase < 1.0:
            return 0

        self._pwm_phase -= 1.0
        return button_mask


def _threshold_drive_mask(drive: float, *, deadzone: float) -> int:
    # The SAC experiment keeps brake out of the action space: negative drive coasts.
    return THROTTLE_MASK if drive > deadzone else 0


def _pwm_duty_cycle(drive: float, *, deadzone: float) -> float:
    # Uses the full continuous policy range for throttle duty: -1 coast, +1 full throttle.
    duty = (drive + 1.0) * 0.5
    if duty <= deadzone:
        return 0.0
    return (duty - deadzone) / (1.0 - deadzone)


def _continuous_action_array(
    action: ActionValue,
    *,
    expected_size: int,
    action_label: str,
    field_labels: tuple[str, ...],
) -> np.ndarray:
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


def _discrete_action_array(
    action: ActionBranchValue,
    *,
    expected_size: int,
    action_label: str,
    field_labels: tuple[str, ...],
) -> np.ndarray:
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


def _hybrid_branch(
    action: Mapping[str, ActionBranchValue],
    branch_name: str,
) -> ActionBranchValue:
    try:
        return action[branch_name]
    except KeyError as exc:
        raise ValueError(f"Hybrid action is missing {branch_name!r} branch") from exc
