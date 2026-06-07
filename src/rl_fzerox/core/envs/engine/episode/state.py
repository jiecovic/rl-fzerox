# src/rl_fzerox/core/envs/engine/episode/state.py
from __future__ import annotations

from dataclasses import dataclass, field

from fzerox_emulator import FZeroXTelemetry, RaceControlState

from ..reset import SelectedTrack


@dataclass(slots=True)
class EngineEpisodeState:
    """Mutable state that spans one active episode in the env engine."""

    active_track: SelectedTrack | None = None
    done: bool = False
    uses_custom_baseline: bool = False
    frame_count: int = 0
    stalled_steps: int = 0
    progress_frontier_stalled_frames: int = 0
    progress_frontier_distance: float = 0.0
    progress_frontier_initialized: bool = False
    return_value: float = 0.0
    boost_pad_entries: int = 0
    airborne_frames: int = 0
    held_control_state: RaceControlState = field(default_factory=RaceControlState)
    last_requested_control_state: RaceControlState = field(default_factory=RaceControlState)
    last_gas_level: float = 0.0
    last_info: dict[str, object] = field(default_factory=dict)
    last_telemetry: FZeroXTelemetry | None = None

    def begin_reset(self, *, active_track: SelectedTrack | None) -> None:
        """Reset episode-local mutable fields before a new initial observation."""

        self.active_track = active_track
        self.done = False
        self.frame_count = 0
        self.stalled_steps = 0
        self.progress_frontier_stalled_frames = 0
        self.progress_frontier_distance = 0.0
        self.progress_frontier_initialized = False
        self.return_value = 0.0
        self.boost_pad_entries = 0
        self.airborne_frames = 0
        self.held_control_state = RaceControlState()
        self.last_requested_control_state = RaceControlState()
        self.last_gas_level = 0.0

    def record_step(
        self,
        *,
        telemetry: FZeroXTelemetry | None,
        requested_control_state: RaceControlState,
        gas_level: float,
        frame_count: int,
        stalled_steps: int,
        progress_frontier_stalled_frames: int,
        progress_frontier_distance: float,
        progress_frontier_initialized: bool,
        return_value: float,
        boost_pad_entries: int,
        airborne_frames: int,
        done: bool,
        info: dict[str, object],
    ) -> None:
        """Persist the episode fields produced by one assembled env step."""

        self.last_telemetry = telemetry
        self.last_requested_control_state = requested_control_state
        self.last_gas_level = gas_level
        self.frame_count = frame_count
        self.stalled_steps = stalled_steps
        self.progress_frontier_stalled_frames = progress_frontier_stalled_frames
        self.progress_frontier_distance = progress_frontier_distance
        self.progress_frontier_initialized = progress_frontier_initialized
        self.return_value = return_value
        self.boost_pad_entries = boost_pad_entries
        self.airborne_frames = airborne_frames
        self.done = done
        self.last_info = dict(info)
