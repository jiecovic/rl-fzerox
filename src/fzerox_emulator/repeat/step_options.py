# src/fzerox_emulator/repeat/step_options.py
"""Python-side configuration for one native repeated-step call."""

from __future__ import annotations

from dataclasses import dataclass

from fzerox_emulator.control.spin import SpinRequest


@dataclass(frozen=True, slots=True)
class RepeatStepConfig:
    """Stop-condition and action-repeat settings for native frame stepping."""

    action_repeat: int
    stuck_min_speed_kph: float
    energy_loss_epsilon: float
    max_episode_steps: int
    progress_frontier_stall_limit_frames: int | None
    progress_frontier_epsilon: float
    terminate_on_energy_depleted: bool
    lean_timer_assist: bool
    spin_request: SpinRequest = "none"
