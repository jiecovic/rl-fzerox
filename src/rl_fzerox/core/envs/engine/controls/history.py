# src/rl_fzerox/core/envs/engine/controls/history.py
from __future__ import annotations

from dataclasses import dataclass, field

from fzerox_emulator import ControllerState
from rl_fzerox.core.domain.lean import DEFAULT_LEAN_MODE, LeanMode
from rl_fzerox.core.envs.actions import BOOST_MASK
from rl_fzerox.core.envs.engine.controls.action_history import ActionHistoryBuffer, clamp
from rl_fzerox.core.envs.engine.controls.boost import BoostTimingState
from rl_fzerox.core.envs.engine.controls.lean import LeanControlState
from rl_fzerox.core.envs.engine.controls.pressure import RecentControlPressure
from rl_fzerox.core.envs.observations import (
    OBSERVATION_STATE_DEFAULTS,
    ActionHistoryControl,
)


@dataclass(slots=True)
class ControlStateTracker:
    """Track recent action-derived state that the policy also observes."""

    lean_mode: LeanMode = DEFAULT_LEAN_MODE
    lean_initial_lockout_frames: int = 0
    boost_decision_interval_frames: int = 1
    boost_request_lockout_frames: int = 0
    action_history_len: int | None = OBSERVATION_STATE_DEFAULTS.action_history_len
    action_history_controls: tuple[ActionHistoryControl, ...] = (
        OBSERVATION_STATE_DEFAULTS.action_history_controls
    )
    _left_steer_held: bool = False
    _right_steer_held: bool = False
    _lean: LeanControlState = field(init=False)
    _boost: BoostTimingState = field(init=False)
    _pressure: RecentControlPressure = field(init=False)
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
        self._pressure = RecentControlPressure()
        self._action_history = ActionHistoryBuffer(
            length=self.action_history_len,
            controls=self.action_history_controls,
        )

    def reset(self) -> None:
        """Clear all step-to-step control history."""

        self._left_steer_held = False
        self._right_steer_held = False
        self._lean.reset()
        self._boost.reset()
        self._pressure.reset()
        self._action_history.reset()

    def apply_lean_semantics(self, control_state: ControllerState) -> ControllerState:
        """Apply the selected Z/R lean primitive semantics to one action."""

        return self._lean.apply_semantics(control_state)

    def record_step(
        self,
        *,
        control_state: ControllerState,
        frames_run: int,
        gas_level: float | None = None,
    ) -> None:
        """Advance tracked control history by one env step."""

        frames_elapsed = max(int(frames_run), 0)
        boost_requested = bool(control_state.joypad_mask & BOOST_MASK)
        steer_axis = clamp(float(control_state.left_stick_x), -1.0, 1.0)

        self._pressure.record_boost(requested=boost_requested, frames_run=frames_elapsed)
        self._pressure.record_steer(axis=steer_axis, frames_run=frames_elapsed)
        self._boost.record(boost_requested=boost_requested, frames_elapsed=frames_elapsed)
        self._lean.record(joypad_mask=control_state.joypad_mask, frames_elapsed=frames_elapsed)
        self._left_steer_held = steer_axis < -1.0e-6
        self._right_steer_held = steer_axis > 1.0e-6
        self._action_history.record(control_state, gas_level=gas_level)

    def observation_fields(self) -> dict[str, float]:
        """Return control-history features passed into observation building."""

        return {
            **self._lean.observation_fields(),
            "recent_boost_pressure": self._pressure.boost_pressure(),
            "steer_left_held": float(self._left_steer_held),
            "steer_right_held": float(self._right_steer_held),
            "recent_steer_pressure": self._pressure.steer_pressure(),
        }

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
