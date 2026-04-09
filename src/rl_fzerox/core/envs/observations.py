# src/rl_fzerox/core/envs/observations.py
from __future__ import annotations

from typing import Literal, TypeAlias

import numpy as np
from gymnasium import spaces

from fzerox_emulator import FZeroXTelemetry, ObservationSpec

ObservationMode: TypeAlias = Literal["image", "image_state"]
ImageObservation: TypeAlias = np.ndarray
ImageStateObservation: TypeAlias = dict[str, np.ndarray]
ObservationValue: TypeAlias = ImageObservation | ImageStateObservation

STATE_FEATURE_NAMES: tuple[str, ...] = (
    "speed_norm",
    "energy_frac",
    "reverse_active",
    "airborne",
    "can_boost",
)
STATE_FEATURE_COUNT = len(STATE_FEATURE_NAMES)
STATE_SPEED_NORMALIZER_KPH = 1_500.0
STATE_FEATURE_HIGH = np.array([2.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)


def build_observation(
    *,
    image: np.ndarray,
    telemetry: FZeroXTelemetry | None,
    mode: ObservationMode,
) -> ObservationValue:
    if mode == "image":
        return image
    if mode == "image_state":
        return {
            "image": image,
            "state": telemetry_state_vector(telemetry),
        }
    raise ValueError(f"Unsupported observation mode: {mode!r}")


def telemetry_state_vector(telemetry: FZeroXTelemetry | None) -> np.ndarray:
    """Build the normalized scalar policy-state vector from live game telemetry."""

    if telemetry is None:
        return np.zeros(STATE_FEATURE_COUNT, dtype=np.float32)

    player = telemetry.player
    max_energy = float(player.max_energy)
    energy_frac = 0.0 if max_energy <= 0.0 else float(player.energy) / max_energy
    return np.array(
        [
            _clamp(float(player.speed_kph) / STATE_SPEED_NORMALIZER_KPH, 0.0, 2.0),
            _clamp(energy_frac, 0.0, 1.0),
            1.0 if player.reverse_timer > 0 else 0.0,
            1.0 if player.airborne else 0.0,
            1.0 if player.can_boost else 0.0,
        ],
        dtype=np.float32,
    )


def build_observation_space(
    observation_spec: ObservationSpec,
    *,
    frame_stack: int,
    mode: ObservationMode,
) -> spaces.Box | spaces.Dict:
    image_space = build_image_observation_space(observation_spec, frame_stack=frame_stack)
    if mode == "image":
        return image_space
    if mode == "image_state":
        return spaces.Dict(
            {
                "image": image_space,
                "state": spaces.Box(
                    low=np.zeros(STATE_FEATURE_COUNT, dtype=np.float32),
                    high=STATE_FEATURE_HIGH,
                    shape=(STATE_FEATURE_COUNT,),
                    dtype=np.float32,
                ),
            }
        )
    raise ValueError(f"Unsupported observation mode: {mode!r}")


def build_image_observation_space(
    observation_spec: ObservationSpec,
    *,
    frame_stack: int,
) -> spaces.Box:
    return spaces.Box(
        low=0,
        high=255,
        shape=image_observation_shape(observation_spec, frame_stack=frame_stack),
        dtype=np.uint8,
    )


def image_observation_shape(
    observation_spec: ObservationSpec,
    *,
    frame_stack: int,
) -> tuple[int, int, int]:
    return (
        observation_spec.height,
        observation_spec.width,
        observation_spec.channels * frame_stack,
    )


def observation_image(observation: ObservationValue) -> np.ndarray:
    if isinstance(observation, dict):
        image = observation.get("image")
        if not isinstance(image, np.ndarray):
            raise ValueError("Dict observation is missing ndarray key 'image'")
        return image
    return observation


def observation_state(observation: ObservationValue) -> np.ndarray | None:
    if not isinstance(observation, dict):
        return None
    state = observation.get("state")
    if state is None:
        return None
    if not isinstance(state, np.ndarray):
        raise ValueError("Dict observation key 'state' must be an ndarray")
    return state


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))
