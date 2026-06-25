# src/rl_fzerox/core/envs/engine/controls/history.py
"""Frame-local control-state tracker for runtime semantics.

The tracker remembers applied and requested controls across steps so pulse,
hold, action-history, and mask override behavior can be computed consistently.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from fzerox_emulator import RaceControlState
from rl_fzerox.core.domain.race import DEFAULT_LEAN_MODE, LeanMode
from rl_fzerox.core.envs.engine.controls.action_history import ActionHistoryBuffer
from rl_fzerox.core.envs.engine.controls.air_brake import AirBrakePulseState
from rl_fzerox.core.envs.engine.controls.boost import BoostTimingState
from rl_fzerox.core.envs.engine.controls.lean import LeanControlState
from rl_fzerox.core.envs.observations import ActionHistoryControl


@dataclass(slots=True)
class ControlStateTracker:
    """Track recent action-derived state that the policy also observes."""

    lean_mode: LeanMode = DEFAULT_LEAN_MODE
    lean_initial_lockout_frames: int = 0
    boost_decision_interval_frames: int = 1
    boost_request_lockout_frames: int = 0
    air_brake_pulse_frames: int = 0
    action_history_len: int | None = None
    action_history_controls: tuple[ActionHistoryControl, ...] = ()
    split_lean_history: bool = False
    _lean: LeanControlState = field(init=False)
    _boost: BoostTimingState = field(init=False)
    _air_brake: AirBrakePulseState = field(init=False)
    _action_history: ActionHistoryBuffer = field(init=False)

    def __post_init__(self) -> None:
        self._lean = LeanControlState(
            mode=self.lean_mode,
            initial_lockout_frames=self.lean_initial_lockout_frames,
        )
        self._boost = BoostTimingState(
            decision_interval_frames=self.boost_decision_interval_frames,
            request_lockout_frames=self.boost_request_lockout_frames,
        )
        self._air_brake = AirBrakePulseState(pulse_frames=self.air_brake_pulse_frames)
        self._action_history = ActionHistoryBuffer(
            length=self.action_history_len,
            controls=self.action_history_controls,
            split_lean_history=self.split_lean_history,
        )

    def reset(self) -> None:
        """Clear all step-to-step control history."""

        self._lean.reset()
        self._boost.reset()
        self._air_brake.reset()
        self._action_history.reset()

    def apply_lean_semantics(self, control_state: RaceControlState) -> RaceControlState:
        """Apply the selected Z/R lean primitive semantics to one action."""

        return self._lean.apply_semantics(control_state)

    def apply_air_brake_semantics(
        self,
        control_state: RaceControlState,
        *,
        available: bool,
    ) -> RaceControlState:
        """Apply fixed-pulse air-brake semantics to one action."""

        return self._air_brake.apply_semantics(control_state, available=available)

    def record_step(
        self,
        *,
        control_state: RaceControlState,
        requested_control_state: RaceControlState | None = None,
        frames_run: int,
        gas_level: float | None = None,
    ) -> None:
        """Advance tracked control history by one env step."""

        frames_elapsed = max(int(frames_run), 0)
        boost_requested = control_state.boost
        self._boost.record(boost_requested=boost_requested, frames_elapsed=frames_elapsed)
        self._lean.record(control_state=control_state, frames_elapsed=frames_elapsed)
        self._air_brake.record(
            control_state=control_state,
            requested_control_state=(
                control_state if requested_control_state is None else requested_control_state
            ),
            frames_elapsed=frames_elapsed,
        )
        self._action_history.record(
            control_state if requested_control_state is None else requested_control_state,
            gas_level=gas_level,
        )

    def action_history_fields(self) -> dict[str, float]:
        """Return fixed-width previous-action features for policy observations."""

        return self._action_history.fields()

    def boost_action_allowed_by_timing(self) -> bool:
        """Return whether this frame is a tactical manual-boost decision slot."""

        return self._boost.action_allowed()

    def lean_action_mask_override(self) -> tuple[int, ...] | None:
        """Return live lean branch restrictions implied by the selected mode."""

        return self._lean.action_mask_override(
            episode_frame_index=self._boost.episode_frame_index,
        )

    def air_brake_action_mask_override(self) -> tuple[int, ...] | None:
        """Return live air-brake branch restrictions implied by an active pulse."""

        return self._air_brake.action_mask_override()

    @property
    def air_brake_pulse_active(self) -> bool:
        return self._air_brake.pulse_active

    @property
    def air_brake_pulse_remaining_frames(self) -> int:
        return self._air_brake.pulse_remaining_frames
