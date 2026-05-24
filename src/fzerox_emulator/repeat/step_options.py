# src/fzerox_emulator/repeat/step_options.py
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RepeatStepConfig:
    action_repeat: int
    stuck_min_speed_kph: float
    energy_loss_epsilon: float
    max_episode_steps: int
    progress_frontier_stall_limit_frames: int | None
    progress_frontier_epsilon: float
    terminate_on_energy_depleted: bool
    lean_timer_assist: bool
