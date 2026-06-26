# src/rl_fzerox/core/career_mode/policy/runtime.py
"""Policy handoff models for Career Mode races.

The watch runtime wraps a trained policy with the inference-safe train config
and delegates race control to the shared policy-drive runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from fzerox_emulator import RaceControlState, SpinRequest
from rl_fzerox.core.envs.actions import ActionValue
from rl_fzerox.core.envs.engine.controls import ActionMaskBranches, ActionMaskSnapshot
from rl_fzerox.core.envs.observations import ObservationValue
from rl_fzerox.core.envs.policy_drive import PolicyDriveFrame, PolicyDriveRuntime
from rl_fzerox.core.manager.training import build_managed_train_app_config
from rl_fzerox.core.runtime_spec.inference import inference_train_app_config
from rl_fzerox.core.runtime_spec.schema import TrainAppConfig

if TYPE_CHECKING:
    from fzerox_emulator import Emulator
    from rl_fzerox.core.manager.models import ManagedPolicySource, ManagedSaveCourseSetup
    from rl_fzerox.core.training.inference import PolicyRunner


@dataclass(frozen=True, slots=True)
class CareerModePolicyControl:
    """Loaded trained policy selected for the current Career Mode race."""

    course_setup: ManagedSaveCourseSetup
    policy_source: ManagedPolicySource
    runner: PolicyRunner

    @property
    def key(self) -> tuple[str, str, str]:
        return (
            self.course_setup.policy_source_kind,
            self.course_setup.policy_source_id,
            self.course_setup.policy_artifact,
        )


class CareerPolicyRaceDriver:
    """Policy adapter for one Career Mode race handoff."""

    def __init__(
        self,
        *,
        emulator: Emulator,
        policy_control: CareerModePolicyControl,
    ) -> None:
        self.policy_control = policy_control
        self.train_config = _policy_train_config(policy_control)
        self._runtime = PolicyDriveRuntime(
            emulator=emulator,
            train_config=self.train_config,
        )

    @property
    def key(self) -> tuple[str, str, str]:
        return self.policy_control.key

    @property
    def last_requested_control_state(self) -> RaceControlState:
        return self._runtime.last_requested_control_state

    @property
    def last_gas_level(self) -> float:
        return self._runtime.last_gas_level

    def begin(
        self,
        *,
        seed: int | None,
        course_id: str | None,
    ) -> tuple[ObservationValue, dict[str, object]]:
        observation, info = self._runtime.begin(
            seed=seed,
            course_id=course_id,
        )
        return observation, info

    def step_policy(
        self,
        action: ActionValue,
        *,
        capture_audio: bool = False,
    ) -> PolicyDriveFrame:
        return self._runtime.step_policy(action, capture_audio=capture_audio)

    def step_manual(
        self,
        control_state: RaceControlState,
        *,
        spin_request: SpinRequest = "none",
        capture_audio: bool = False,
    ) -> PolicyDriveFrame:
        return self._runtime.step_manual(
            control_state,
            spin_request=spin_request,
            capture_audio=capture_audio,
        )

    def action_mask_branches(self) -> ActionMaskBranches:
        return self._runtime.action_mask_branches()

    def action_mask_snapshot(self) -> ActionMaskSnapshot:
        return self._runtime.action_mask_snapshot()


def _policy_train_config(policy_control: CareerModePolicyControl) -> TrainAppConfig:
    train_config = build_managed_train_app_config(
        policy_control.policy_source.config,
        run_id=policy_control.policy_source.id,
        run_dir=policy_control.policy_source.source_dir,
    )
    return inference_train_app_config(train_config)
