# src/rl_fzerox/ui/watch/runtime/ipc/messages.py
from __future__ import annotations

from dataclasses import dataclass

from fzerox_emulator import RaceControlState
from fzerox_emulator.arrays import ObservationFrame, RgbFrame, StateVector
from rl_fzerox.core.envs.actions import ActionValue
from rl_fzerox.core.envs.engine.controls import ActionMaskBranches
from rl_fzerox.core.runtime_spec.schema import TrackSamplingConfig
from rl_fzerox.ui.watch.runtime.cnn import (
    DEFAULT_CNN_ACTIVATION_NORMALIZATION,
    CnnActivationNormalizationMode,
    CnnActivationSnapshot,
)


@dataclass(frozen=True)
class ViewerCommand:
    """One UI command sent from pygame to the simulation process."""

    quit_requested: bool = False
    toggle_pause: bool = False
    step_once: bool = False
    save_state: bool = False
    reset_mode: str | None = None
    jump_course_id: str | None = None
    toggle_deterministic_policy: bool = False
    toggle_manual_control: bool = False
    toggle_current_course_lock: bool = False
    toggle_zeroed_state_feature_name: str | None = None
    control_fps_delta: int = 0
    reset_control_fps: bool = False
    cnn_visualization_enabled: bool = False
    auxiliary_visualization_enabled: bool = False
    cnn_normalization: CnnActivationNormalizationMode = DEFAULT_CNN_ACTIVATION_NORMALIZATION
    control_state: RaceControlState | None = None


@dataclass(frozen=True)
class WorkerCommandBatch:
    """Coalesced commands consumed by the simulation process."""

    quit_requested: bool
    paused: bool
    step_requests: int
    save_requests: int
    reset_mode: str | None
    jump_course_id: str | None
    toggle_deterministic_policy: bool
    manual_control_enabled: bool
    toggle_current_course_lock: bool
    toggle_zeroed_state_feature_name: str | None
    control_fps_delta: int
    reset_control_fps: bool
    cnn_visualization_enabled: bool
    auxiliary_visualization_enabled: bool
    cnn_normalization: CnnActivationNormalizationMode
    control_state: RaceControlState


@dataclass(frozen=True)
class WorkerError:
    message: str


@dataclass(frozen=True)
class WorkerClosed:
    """Marker published when the simulation worker exits."""


@dataclass(frozen=True)
class WatchSnapshot:
    """Pickle-safe frame and HUD payload published by the simulation process."""

    raw_frame: RgbFrame
    observation_image: ObservationFrame
    observation_state: StateVector | None
    observation_state_reference: StateVector | None
    info: dict[str, object]
    reset_info: dict[str, object]
    episode: int
    episode_reward: float
    control_fps: float
    target_control_fps: float | None
    native_fps: float
    control_state: RaceControlState
    gas_level: float
    boost_lamp_level: float
    action_mask_branches: ActionMaskBranches
    policy_action: ActionValue | None
    policy_label: str | None
    policy_curriculum_stage: str | None
    policy_num_timesteps: int | None
    policy_experience_frames: int | None
    policy_deterministic: bool | None
    manual_control_enabled: bool
    policy_reload_age_seconds: float | None
    policy_reload_error: str | None
    cnn_activations: CnnActivationSnapshot | None
    best_finish_position: int | None
    best_finish_ranks: dict[str, int]
    best_finish_times: dict[str, int]
    latest_finish_times: dict[str, int]
    latest_finish_deltas_ms: dict[str, int]
    failed_track_attempts: frozenset[str]
    continuous_air_brake_disabled: bool
    telemetry_data: dict[str, object] | None
    active_track_sampling: TrackSamplingConfig | None = None
    policy_auxiliary_state_predictions: dict[str, object] | None = None
    policy_auxiliary_state_targets: dict[str, object] | None = None
    action_hold_frame: int = 1
    action_hold_frames: int = 1
    policy_decision_frame: bool = True
