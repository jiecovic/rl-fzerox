# src/rl_fzerox/ui/watch/runtime/career_mode/loop/policy_runtime.py
"""Policy-control runtime values for the Career Mode watch loop.

The Career Mode controller owns lifecycle decisions: when an attempt starts,
when policy control is allowed, and when a terminal result is observed. This
module only groups worker-local values that describe the active policy rollout
for snapshots, manual control, CNN/Aux visualizations, and policy stepping.
"""

from __future__ import annotations

from dataclasses import dataclass

from fzerox_emulator import RaceControlState
from rl_fzerox.core.career_mode.policy import CareerModePolicyControl
from rl_fzerox.core.envs.actions import ActionValue
from rl_fzerox.ui.watch.runtime.career_mode.loop.state import CareerModeLoopState
from rl_fzerox.ui.watch.runtime.career_mode.policy_step import CareerPolicyStepResult
from rl_fzerox.ui.watch.runtime.policy.cnn import CnnActivationSnapshot


@dataclass(slots=True)
class CareerPolicyRuntime:
    """Mutable policy-rollout state owned by the worker loop."""

    policy_control: CareerModePolicyControl | None
    started: bool
    manual_enabled: bool
    policy_action: ActionValue | None
    control_state: RaceControlState
    gas_level: float
    boost_lamp_level: float
    episode_reward: float
    cnn_activations: CnnActivationSnapshot | None
    auxiliary_predictions: dict[str, object] | None
    auxiliary_targets: dict[str, object] | None

    @classmethod
    def from_loop_state(cls, state: CareerModeLoopState) -> CareerPolicyRuntime:
        return cls(
            policy_control=state.active_policy_control,
            started=state.active_policy_started,
            manual_enabled=state.manual_control_enabled,
            policy_action=state.current_policy_action,
            control_state=state.current_control_state,
            gas_level=state.current_gas_level,
            boost_lamp_level=state.boost_lamp_level,
            episode_reward=state.episode_reward,
            cnn_activations=state.cnn_activations,
            auxiliary_predictions=state.current_auxiliary_predictions,
            auxiliary_targets=state.current_auxiliary_targets,
        )

    def clear(self, *, reset_episode_reward: bool = False) -> None:
        """Drop active policy rollout state after menu or terminal transitions."""

        self.policy_control = None
        self.started = False
        self.manual_enabled = False
        self.policy_action = None
        self.control_state = RaceControlState()
        self.gas_level = 0.0
        self.boost_lamp_level = 0.0
        self.cnn_activations = None
        self.auxiliary_predictions = None
        self.auxiliary_targets = None
        if reset_episode_reward:
            self.episode_reward = 0.0

    def reset_menu_controls(self) -> None:
        """Reset visible race controls while preserving pending policy control."""

        self.episode_reward = 0.0
        self.control_state = RaceControlState()
        self.gas_level = 0.0
        self.boost_lamp_level = 0.0

    def mark_policy_race_started(
        self,
        *,
        episode_reward: float,
        control_state: RaceControlState,
        gas_level: float,
    ) -> None:
        """Publish the initial runtime values after entering policy race control."""

        self.episode_reward = episode_reward
        self.policy_action = None
        self.control_state = control_state
        self.gas_level = gas_level
        self.auxiliary_predictions = None
        self.auxiliary_targets = None
        self.started = True

    def apply_policy_step(self, policy_step: CareerPolicyStepResult) -> None:
        """Update runtime values from one completed policy/manual step."""

        self.episode_reward = policy_step.episode_reward
        self.control_state = policy_step.control_state
        self.gas_level = policy_step.gas_level
        self.boost_lamp_level = policy_step.boost_lamp_level
        self.policy_action = policy_step.policy_action
        self.cnn_activations = policy_step.cnn_activations
        self.auxiliary_predictions = policy_step.auxiliary_predictions
        self.auxiliary_targets = policy_step.auxiliary_targets
