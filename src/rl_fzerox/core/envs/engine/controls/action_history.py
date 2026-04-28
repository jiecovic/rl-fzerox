# src/rl_fzerox/core/envs/engine/controls/action_history.py
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from fzerox_emulator import ControllerState
from rl_fzerox.core.envs.actions import (
    ACCELERATE_MASK,
    AIR_BRAKE_MASK,
    BOOST_MASK,
)
from rl_fzerox.core.envs.engine.controls.lean import lean_index_from_mask, signed_lean
from rl_fzerox.core.envs.observations import (
    OBSERVATION_STATE_DEFAULTS,
    ActionHistoryControl,
)


@dataclass(frozen=True, slots=True)
class ActionHistorySample:
    steer: float
    gas: float
    air_brake: float
    boost: float
    lean: float
    pitch: float


@dataclass(slots=True)
class ActionHistoryBuffer:
    """Fixed-width previous-action observation feature buffer."""

    length: int | None = OBSERVATION_STATE_DEFAULTS.action_history_len
    controls: tuple[ActionHistoryControl, ...] = OBSERVATION_STATE_DEFAULTS.action_history_controls
    _resolved_length: int = field(init=False, default=0)
    _samples: deque[ActionHistorySample] = field(init=False)

    def __post_init__(self) -> None:
        self._resolved_length = _resolve_action_history_len(self.length)
        self._samples = deque(maxlen=self._resolved_length)

    def reset(self) -> None:
        self._samples.clear()

    def record(self, control_state: ControllerState, *, gas_level: float | None) -> None:
        joypad = control_state.joypad_mask
        normalized_gas = (
            (1.0 if joypad & ACCELERATE_MASK else 0.0)
            if gas_level is None
            else clamp(float(gas_level), 0.0, 1.0)
        )
        self._samples.append(
            ActionHistorySample(
                steer=clamp(float(control_state.left_stick_x), -1.0, 1.0),
                gas=normalized_gas,
                air_brake=1.0 if joypad & AIR_BRAKE_MASK else 0.0,
                boost=1.0 if joypad & BOOST_MASK else 0.0,
                lean=signed_lean(lean_index_from_mask(joypad)),
                pitch=clamp(float(control_state.left_stick_y), -1.0, 1.0),
            )
        )

    def fields(self) -> dict[str, float]:
        samples = list(reversed(self._samples))
        fields: dict[str, float] = {}
        for index in range(self._resolved_length):
            sample = samples[index] if index < len(samples) else _empty_sample()
            suffix = index + 1
            fields[f"prev_steer_{suffix}"] = sample.steer
            fields[f"prev_gas_{suffix}"] = sample.gas
            fields[f"prev_air_brake_{suffix}"] = sample.air_brake
            fields[f"prev_boost_{suffix}"] = sample.boost
            fields[f"prev_lean_{suffix}"] = sample.lean
            fields[f"prev_pitch_{suffix}"] = sample.pitch
        return {
            key: value for key, value in fields.items() if _field_control(key) in self.controls
        }


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _empty_sample() -> ActionHistorySample:
    return ActionHistorySample(
        steer=0.0,
        gas=0.0,
        air_brake=0.0,
        boost=0.0,
        lean=0.0,
        pitch=0.0,
    )


def _resolve_action_history_len(action_history_len: int | None) -> int:
    if action_history_len is None:
        return 0
    length = int(action_history_len)
    if length <= 0:
        raise ValueError("action_history_len must be positive or None")
    return length


def _field_control(field_name: str) -> ActionHistoryControl:
    control_name = field_name.removeprefix("prev_").rsplit("_", maxsplit=1)[0]
    if control_name == "steer":
        return "steer"
    if control_name == "gas":
        return "gas"
    if control_name == "air_brake":
        return "air_brake"
    if control_name == "boost":
        return "boost"
    if control_name == "lean":
        return "lean"
    if control_name == "pitch":
        return "pitch"
    raise ValueError(f"Unsupported action history field: {field_name!r}")
