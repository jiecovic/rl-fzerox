# src/rl_fzerox/ui/watch/runtime/ipc/messages.py
from __future__ import annotations

from dataclasses import dataclass

from fzerox_emulator import ControllerState
from fzerox_emulator.arrays import ObservationFrame, RgbFrame, StateVector
from rl_fzerox.core.envs.actions import ActionValue
from rl_fzerox.core.envs.engine.controls import ActionMaskBranches


@dataclass(frozen=True)
class ViewerCommand:
    """One UI command sent from pygame to the simulation process."""

    quit_requested: bool = False
    toggle_pause: bool = False
    step_once: bool = False
    save_state: bool = False
    force_reset: bool = False
    toggle_deterministic_policy: bool = False
    control_fps_delta: int = 0
    control_state: ControllerState | None = None


@dataclass(frozen=True)
class WorkerCommandBatch:
    """Coalesced commands consumed by the simulation process."""

    quit_requested: bool
    paused: bool
    step_requests: int
    save_requests: int
    reset_requested: bool
    toggle_deterministic_policy: bool
    control_fps_delta: int
    control_state: ControllerState


@dataclass(frozen=True)
class WorkerError:
    message: str


@dataclass(frozen=True)
class WorkerClosed:
    pass


@dataclass(frozen=True)
class WatchSnapshot:
    """Pickle-safe frame and HUD payload published by the simulation process."""

    raw_frame: RgbFrame
    observation_image: ObservationFrame
    observation_state: StateVector | None
    info: dict[str, object]
    reset_info: dict[str, object]
    episode: int
    episode_reward: float
    control_fps: float
    target_control_fps: float | None
    native_fps: float
    control_state: ControllerState
    gas_level: float
    boost_lamp_level: float
    action_mask_branches: ActionMaskBranches
    policy_action: ActionValue | None
    policy_label: str | None
    policy_curriculum_stage: str | None
    policy_num_timesteps: int | None
    policy_deterministic: bool | None
    policy_reload_age_seconds: float | None
    policy_reload_error: str | None
    best_finish_position: int | None
    best_finish_times: dict[str, int]
    latest_finish_times: dict[str, int]
    latest_finish_deltas_ms: dict[str, int]
    continuous_air_brake_disabled: bool
    telemetry_data: dict[str, object] | None
    action_hold_frame: int = 1
    action_hold_frames: int = 1
    policy_decision_frame: bool = True
